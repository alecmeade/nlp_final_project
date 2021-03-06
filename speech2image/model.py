import os
import copy
import numpy
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .espnet_encoder import ESPnetEncoder
from .networks import Encoder, Generator, Discriminator
from .augment import augment, AdaptiveAugment
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
ADA_INTERVAL = 4
ADA_KIMG = 100
ADA_TARGET = 0.6


class Speech2Image(pl.LightningModule):
    def __init__(self, img_size=256, latent=512, n_mlp=8, pretrained=None):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self.s_flag = False
        self.p = 0.0
        self.rtstat = 0
        self.aug = AdaptiveAugment(ADA_TARGET, ADA_KIMG, ADA_INTERVAL)
        self._init_networks(img_size, self.latent_size, n_mlp, pretrained)

    def _init_networks(self, img_size, latent, n_mlp, pretrained=None):
        self.enc = ESPnetEncoder.from_pretrained()
        self.enc.train()
        self.G = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        self.D = Discriminator(img_size, channel_multiplier=CHANNEL_MULTIPLIER)
        self.G_EMA = copy.deepcopy(self.G).eval()
        self.G.train()
        self.D.train()
        accumulate(self.G_EMA, self.G, 0)
        if pretrained:
            print("Loading pretrained from %s." % pretrained)
            c = torch.load(pretrained)
            self.G.load_state_dict(c["g"], strict=False)
            self.G_EMA.load_state_dict(c["g_ema"], strict=False)

    def forward(self, x=None, nframes=None):
        x = self.enc(x)
        z = x.mean(dim=1).view(1, x.shape[0], -1)
        z = torch.cat([z, torch.randn(1, x.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        return self.G_EMA(z, randomize_noise=False)[0]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.s_flag:
            accumulate(self.G_EMA, self.G, ACCUM)
        else:
            self.s_flag = True
        images, audio, nframes, apath = batch

        # Set up
        g_step = optimizer_idx == 0
        d_step = optimizer_idx == 1
        enc_step = optimizer_idx == 2

        # Generate image
        x = self.enc(audio)
        z = x.mean(dim=1).view(1, audio.shape[0], -1)
        z = torch.cat([z, torch.randn(1, audio.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        fake_imgs, _ = self.G(z, randomize_noise=False)

        rimgs = images.clone()
        images, _ = augment(images, self.p)
        fake_imgs, _ = augment(fake_imgs, self.p)

        fake_pred = self.D(fake_imgs)
        real_pred = self.D(images)
        
        self.p = self.aug.tune(real_pred)
        self.rtstat = self.aug.r_t_stat

        self.log("RT", self.rtstat, on_step=True, on_epoch=True, prog_bar=True)

        if d_step:
            d_loss = d_logistic_loss(real_pred, fake_pred)
            self.log("D", d_loss, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx % D_REG == 0:
                rimgs.requires_grad = True
                images, _ = augment(rimgs, self.p)
                real_pred = self.D(images)
                r1_loss = d_r1_loss(real_pred, rimgs)
                self.log("D_R1", r1_loss, on_step=True, on_epoch=True, prog_bar=True)

                return d_loss + (R1 / 2 * r1_loss * D_REG + 0 * real_pred[0])
            return d_loss
        elif g_step or enc_step:
            fake_pred = self.D(fake_imgs)
            g_loss = g_nonsaturating_loss(fake_pred)
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
        self.g_optim = optim.Adam(list(self.G.parameters()) + list(self.enc.parameters()), lr=G_LR, betas=ADAM_BETAG)
        self.d_optim = optim.Adam(self.D.parameters(), lr=D_LR, betas=ADAM_BETAD)
        self.enc_optim = optim.Adam(self.enc.parameters(), lr=ENC_LR, betas=ADAM_BETAG)
        return [self.g_optim, self.d_optim, self.enc_optim], []
    
    def validation_step(self, batch, batch_idx):
        images, audio, nframes, apath = batch
        fake_imgs = self.forward(audio, nframes).cpu()
        return {"G_IMGs": fake_imgs, "I_AUDs": apath, "R_IMGs": images.cpu()}
    
    def validation_epoch_end(self, outputs):
        if not len(outputs):
            return
        f_imgs = []
        r_imgs = []
        i_auds = []
        for output in outputs:
            f_imgs += [wandb.Image(x, caption="G_IMG %d" % i) for i, x in enumerate(output["G_IMGs"])]
            r_imgs += [wandb.Image(x, caption="R_IMG %d" % i) for i, x in enumerate(output["R_IMGs"])]
            i_auds += [wandb.Audio(x, caption="I_AUD %d" % i) for i, x in enumerate(output["I_AUDs"])]
        
        self.logger.experiment.log({"G_IMG Val": f_imgs}, commit=False)
        self.logger.experiment.log({"R_IMG Val": r_imgs}, commit=False)
        self.logger.experiment.log({"I_AUD Val": i_auds}, commit=False)

    def test_step(self, batch, batch_idx):
        images, audio, nframes, apath = batch
        fake_imgs = self.forward(audio, nframes).cpu()
        return {"G_IMGs": fake_imgs, "I_AUDs": apath, "R_IMGs": images.cpu()}
        
    def test_epoch_end(self, outputs):
        return outputs
