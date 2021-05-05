import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .networks import Encoder, SimpleGenerator, SimpleDiscriminator
from .util import *


# Hyperparameters
G_LR = 2e-2
D_LR = 2e-4
ADAM_BETA = (0.0, 0.99)


class Speech2Image(pl.LightningModule):
    def __init__(self, latent=512):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self._init_networks(self.latent_size)

    def _init_networks(self, latent):
        self.enc = Encoder(latent_dim=latent//2)
        self.enc.enc.train()
        self.G = SimpleGenerator()
        self.D = SimpleDiscriminator()

    def forward(self, x=None, nframes=None):
        z = torch.cat([self.enc(x, nframes).unsqueeze(0), torch.randn(1, 1, self.latent_size//2, device=self.device)], dim=-1) if self.enc is not None else torch.randn(1, 1, self.latent_size, device=self.device)
        z = z.view(z.shape[0], -1, 1, 1)
        return self.G(z)


    def training_step(self, batch, batch_idx, optimizer_idx):
        images, audio, nframes = batch

        # Set up
        g_step = optimizer_idx == 0
        d_step = optimizer_idx == 1

        # Generate image
        z = torch.cat([self.enc(audio, nframes).unsqueeze(0), torch.randn(1, images.shape[0], self.latent_size//2, device=self.device)], dim=-1) if self.enc is not None else torch.randn(1, images.shape[0], self.latent_size, device=self.device)
        z = z.view(z.shape[0], -1, 1, 1)
        fake_imgs = self.G(z)
        fake_pred = self.D(fake_imgs, z)
        real_pred = self.D(images, z)
        
        self.logger.experiment.log({"G_IMG": [wandb.Image(fake_imgs[0].cpu(), caption="G_IMG"), wandb.Image(images[0].cpu(), caption="R_IMG")]}, commit=False)

        if d_step:
            d_loss = F.mse_loss(fake_pred, torch.zeros(fake_pred.shape, device=self.device))
            d_loss += F.mse_loss(real_pred, torch.ones(real_pred.shape, device=self.device))
            self.log("D", d_loss, on_step=True, on_epoch=True, prog_bar=True)
            return d_loss
        elif g_step:
            g_loss = F.mse_loss(fake_pred, torch.ones(fake_pred.shape, device=self.device))
            g_loss += F.l1_loss(fake_imgs, images) * 100
            self.log("G", g_loss, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss


    def configure_optimizers(self):
        self.g_optim = optim.Adam(list(self.G.parameters()) + list(self.enc.parameters()), lr=G_LR, betas=ADAM_BETA)
        self.d_optim = optim.Adam(self.D.parameters(), lr=D_LR, betas=ADAM_BETA)
        return [self.g_optim, self.d_optim], []
        
    
    def validation_step(self, batch, batch_idx):
        pass
        
    
    def validation_epoch_end(self, outputs):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
        
    
    def test_epoch_end(self, outputs):
        pass
