import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .networks import Encoder, Generator, Discriminator
from .util import *


# Hyperparameters
G_REG = 16
D_REG = 4
R1 = 10
G_LR = 2e-2
D_LR = 2e-2
ENC_LR = 1e-5
ADAM_BETAG = (0.0, 0.99 ** (G_REG/(G_REG + 1)))
ADAM_BETAD = (0.0, 0.99 ** (D_REG/(D_REG + 1)))
PATH_BATCHSHRINK = 2
PATH_REGULARIZE = 2
CHANNEL_MULTIPLIER = 2
ACCUM = 0.5 ** (32 / (10 * 1000))


class Speech2Image(pl.LightningModule):
    def __init__(self, img_size=256, latent=512, n_mlp=8):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self._init_networks(img_size, self.latent_size, n_mlp)

    def _init_networks(self, img_size, latent, n_mlp):
        self.enc = Encoder(latent_dim=latent)
        self.enc.enc.train()
        self.G = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        self.D = Discriminator(img_size, channel_multiplier=CHANNEL_MULTIPLIER)
        self.G_EMA = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        accumulate(self.G_EMA, self.G, 0)

    def forward(self, x=None, nframes=None):
        z = self.enc(x, nframes).unsqueeze(0) if self.enc is not None else torch.randn(1, 1, self.latent_size, device=self.device)
        return self.G_EMA(z)[0]


    def training_step(self, batch, batch_idx, optimizer_idx):
        images, audio, nframes = batch

        # Set up
        g_step = optimizer_idx == 0
        d_step = optimizer_idx == 1

        # Generate image
        z = self.enc(audio, nframes).unsqueeze(0) if self.enc is not None else torch.randn(1, images.shape[0], self.latent_size, device=self.device)
        fake_imgs, _ = self.G(z)
        fake_pred = self.D(fake_imgs)
        real_pred = self.D(images)

        self.logger.experiment.log({"G_IMG": [wandb.Image(fake_imgs[0].cpu(), caption="G_IMG")]}, commit=False)

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
        elif g_step:
            fake_pred = self.D(fake_imgs)
            g_loss = g_nonsaturating_loss(fake_pred)
            self.log("G", g_loss, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx % G_REG == 0:
                path_batchsize = max(1, images.shape[0]//PATH_BATCHSHRINK)
                z = torch.randn(1, path_batchsize, self.latent_size, device=self.device)
                fake_img, latents = self.G(z, return_latents=True)

                path_loss, self.mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
                self.log("PATH", path_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("PATH_L", path_lengths.mean(), on_step=True, on_epoch=True, prog_bar=True)
                return g_loss + (PATH_REGULARIZE * G_REG * path_loss)
            
            accumulate(self.G_EMA, self.G, ACCUM)
            return g_loss


    def configure_optimizers(self):
        self.g_optim = optim.Adam(list(self.G.parameters()) + list(self.enc.parameters()), lr=G_LR, betas=ADAM_BETAG)
        self.d_optim = optim.Adam(self.D.parameters(), lr=D_LR, betas=ADAM_BETAD)
        return [self.g_optim, self.d_optim], []
        
    
    def validation_step(self, batch, batch_idx):
        pass
        
    
    def validation_epoch_end(self, outputs):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
        
    
    def test_epoch_end(self, outputs):
        pass
