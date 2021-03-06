import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .espnet_encoder import ESPnetEncoder
from .networks import SimplestGenerator, SimplestDiscriminator
from .util import *


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ADAM_BETA = (0.0, 0.99)
LAMBDA_GP = 4


class Speech2Image(pl.LightningModule):
    def __init__(self, latent=512):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self._init_networks(self.latent_size)

    def _init_networks(self, latent):
        self.enc = ESPnetEncoder.from_pretrained()
        self.enc.train()
        self.G = SimplestGenerator()
        self.D = SimplestDiscriminator()

    def forward(self, x=None, nframes=None):
        x = self.enc(x)
        z = x.mean(dim=1)
        return self.G(z)


    def training_step(self, batch, batch_idx, optimizer_idx):
        images, audio, nframes, apath = batch

        # Set up
        g_step = optimizer_idx == 0
        d_step = optimizer_idx == 1

        # Generate image
        x = self.enc(audio)
        z = x.mean(dim=1)
        fake_imgs = self.G(z)
        fake_pred = self.D(fake_imgs, z)
        real_pred = self.D(images, z)
        
        self.logger.experiment.log({"G_IMG": [wandb.Image(fake_imgs[0].cpu(), caption="G_IMG"), wandb.Image(images[0].cpu(), caption="R_IMG")]}, commit=False)
        self.logger.experiment.log({"G_AUD": [wandb.Audio(apath[0], caption="Audio")]}, commit=False)
        
        # gp = compute_gradient_penalty(self.D, images.data, fake_imgs.data, l, self.device)

        if d_step:
            d_loss = F.mse_loss(fake_pred, torch.zeros(fake_pred.shape, device=self.device))
            d_loss += F.mse_loss(real_pred, torch.ones(real_pred.shape, device=self.device))
            # d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + LAMBDA_GP * gp
            self.log("D", d_loss, on_step=True, on_epoch=True, prog_bar=True)
            return d_loss
        elif g_step:
            g_loss = F.mse_loss(fake_pred, torch.ones(fake_pred.shape, device=self.device))
            # g_loss = -torch.mean(fake_pred)
            # g_loss += F.l1_loss(fake_imgs, images) * 100
            self.log("G", g_loss, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss


    def configure_optimizers(self):
        self.g_optim = optim.Adam(list(self.G.parameters()) + list(self.enc.parameters()), lr=G_LR, betas=ADAM_BETA)
        self.d_optim = optim.Adam(self.D.parameters(), lr=D_LR, betas=ADAM_BETA)
        return [self.g_optim, self.d_optim], []
        
    
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
        pass
        
    
    def test_epoch_end(self, outputs):
        pass
