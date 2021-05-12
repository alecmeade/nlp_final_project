import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .text_encoder import TextEncoder
from .networks import Encoder, Generator, Discriminator
from .util import *


# Hyperparameters
G_REG = 16
D_REG = 4
G_REGRATIO = (G_REG/(G_REG + 1))
D_REGRATIO = (D_REG/(D_REG + 1))
R1 = 10
G_LR = 2e-3 * G_REGRATIO
D_LR = 4e-4 * D_REGRATIO
ENC_LR = 1e-5
ADAM_BETAG = (0.0, 0.99 ** G_REGRATIO)
ADAM_BETAD = (0.0, 0.99 ** D_REGRATIO)
PATH_BATCHSHRINK = 2
PATH_REGULARIZE = 2
CHANNEL_MULTIPLIER = 2
LAMBDA = 100
ACCUM = 0.5 ** (32 / (10 * 1000))


class Text2Image(pl.LightningModule):
    def __init__(self, img_size=256, latent=512, n_mlp=8, pretrained=None):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self.s_flag = False
        self._init_networks(img_size, self.latent_size, n_mlp, pretrained)

    def _init_networks(self, img_size, latent, n_mlp, pretrained=None):
        self.enc = TextEncoder()
        self.enc.train()
        self.G = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        self.D = Discriminator(img_size, channel_multiplier=CHANNEL_MULTIPLIER)
        self.G_EMA = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        self.G.train()
        self.D.train()
        accumulate(self.G_EMA, self.G, 0)
        if pretrained:
            print("Loading pretrained from %s." % pretrained)
            c = torch.load(pretrained)
            self.G.load_state_dict(c["g"], strict=False)
            self.G_EMA.load_state_dict(c["g_ema"], strict=False)

    def forward(self, x=None):
        x = self.enc(x)
        z = x.view(1, x.shape[0], -1)
        z = torch.cat([z, torch.randn(1, x.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        return self.G_EMA(z, randomize_noise=False)[0]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.s_flag:
            accumulate(self.G_EMA, self.G, ACCUM)
        else:
            self.s_flag = True
        images, text = batch

        # Set up
        g_step = optimizer_idx == 0
        d_step = optimizer_idx == 1
        enc_step = optimizer_idx == 2

        # Generate image
        x = self.enc(text)
        x.requires_grad_(True)
        z = x.view(1, x.shape[0], -1)
        z = torch.cat([z, torch.randn(1, x.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        fake_imgs, _ = self.G(z, randomize_noise=False)
        fake_pred = self.D(fake_imgs, z)
        real_pred = self.D(images, z)

        fake_imgs.requires_grad_(True)
        
        # self.logger.experiment.log({"G_IMG": [wandb.Image(self.forward(audio[0].unsqueeze(0), nframes[0].unsqueeze(0)).cpu(), caption="G_IMG"), wandb.Image(images[0].cpu(), caption="R_IMG")]}, commit=False)
        # self.logger.experiment.log({"G_AUD": [wandb.Audio(apath[0], caption="Audio")]}, commit=False)

        if d_step:
            d_loss = d_logistic_loss(real_pred, fake_pred)
            self.log("D", d_loss, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx % D_REG == 0:
                images.requires_grad = True

                real_pred = self.D(images)
                r1_loss = d_r1_loss(real_pred, images)
                self.log("D_R1", r1_loss, on_step=True, on_epoch=True, prog_bar=True)

                self.D.zero_grad()
                return d_loss + (R1 / 2 * r1_loss * D_REG + 0 * real_pred[0])
            return d_loss
        elif g_step or enc_step:
            fake_pred = self.D(fake_imgs)
            g_loss = g_nonsaturating_loss(fake_pred) + F.l1_loss(fake_imgs, images)
            self.log("G", g_loss, on_step=True, on_epoch=True, prog_bar=True)

            if g_step and (batch_idx % G_REG == 0):
                path_batchsize = max(1, images.shape[0]//PATH_BATCHSHRINK)
                z = torch.randn(1, path_batchsize, self.latent_size, device=self.device)
                fake_img, latents = self.G(z, return_latents=True)

                path_loss, self.mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
                self.log("PATH", path_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("PATH_L", path_lengths.mean(), on_step=True, on_epoch=True, prog_bar=True)
                return g_loss + (PATH_REGULARIZE * G_REG * path_loss)
            return g_loss

    def configure_optimizers(self):
        self.g_optim = optim.Adam(self.G.parameters(), lr=G_LR, betas=ADAM_BETAG)
        self.d_optim = optim.Adam(self.D.parameters(), lr=D_LR, betas=ADAM_BETAD)
        self.enc_optim = optim.Adam(self.enc.parameters(), lr=ENC_LR, betas=ADAM_BETAG)
        return [self.g_optim, self.d_optim, self.enc_optim], []
    
    def validation_step(self, batch, batch_idx):
        images, text = batch
        fake_imgs = self.forward(text).cpu()
        return {"G_IMGs": fake_imgs, "I_TXTs": text, "R_IMGs": images.cpu()}
    
    def validation_epoch_end(self, outputs):
        if not len(outputs):
            return
        f_imgs = []
        r_imgs = []
        i_txts = []
        for output in outputs:
            f_imgs += [wandb.Image(x, caption="G_IMG %d" % i) for i, x in enumerate(output["G_IMGs"])]
            r_imgs += [wandb.Image(x, caption="R_IMG %d" % i) for i, x in enumerate(output["R_IMGs"])]
            i_txts += [wandb.Html(x) for x in output["I_TXTs"]]
        
        self.logger.experiment.log({"G_IMG Val": f_imgs}, commit=False)
        self.logger.experiment.log({"R_IMG Val": r_imgs}, commit=False)
        self.logger.experiment.log({"I_TXT Val": i_txts}, commit=False)

    def test_step(self, batch, batch_idx):
        images, text = batch
        fake_imgs = self.forward(text).cpu()
        return {"G_IMGs": fake_imgs, "I_TXTs": text, "R_IMGs": images.cpu()}
        
    def test_epoch_end(self, outputs):
        return outputs
