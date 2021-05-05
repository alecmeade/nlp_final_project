import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .espnet_encoder import ESPnetEncoder
from dall_e import map_pixels, unmap_pixels, load_model
from vector_quantize_pytorch import VectorQuantize
from .util import *


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ENC_LR = 1e-5
ADAM_BETA = (0.0, 0.99)
LAMBDA_GP = 4


class Speech2Image(pl.LightningModule):
    def __init__(self, latent=512):
        super().__init__()

        self.latent_size = latent
        self.mean_path_length = 0
        self.vq = VectorQuantize(dim=311, n_embed=8192, decay=0.1, commitment=1)
        self.l2 = nn.Linear(512, 1024)
        self.tr = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())
        self._init_networks(self.latent_size)

    def _init_networks(self, latent):
        self.enc = ESPnetEncoder.from_pretrained()
        self.enc.train()
        self.I = load_model("./encoder.pkl")
        self.G = load_model("./decoder.pkl")
        self.G.eval()
        self.I.eval()

    def forward(self, x=None, nframes=None):
        z = torch.cat([self.enc(x, nframes), torch.randn(1, self.latent_size, device=self.device)], dim=-1)
        return self.G(z)


    def training_step(self, batch, batch_idx):
        images, audio, nframes, apath = batch

        # Generate image
        x = self.enc(audio)
        _, l, _= self.vq(self.l2(x).permute(0, 2, 1))
        # z = torch.cat([l, torch.randint(0, 8192, (images.shape[0], 984), device=self.device)], dim=-1).view(-1, 32, 32)
        # z = torch.randint(0, 8192, l.shape, device=self.device)
        k = self.I(map_pixels(images))
        f = torch.argmax(k, dim=1)
        z = F.one_hot(f, num_classes=8192).permute(0, 3, 1, 2).float()
        fake_imgs = unmap_pixels(torch.sigmoid(self.G(z).float()[:, :3]))
        n = F.one_hot(l.view(*list(f.shape)).type_as(f), num_classes=8192).permute(0, 3, 1, 2).float()
        fake_imgs2 = unmap_pixels(torch.sigmoid(self.G(n).float()[:, :3]))
        self.logger.experiment.log({"G_IMG": [wandb.Image(fake_imgs2[0].cpu(), caption="G_IMG"), wandb.Image(fake_imgs[0].cpu(), caption="D_IMG"), wandb.Image(images[0].cpu(), caption="R_IMG")]}, commit=False)
        self.logger.experiment.log({"G_AUD": [wandb.Audio(apath[0], caption="Audio")]}, commit=False)

        j = f.clone().float()
        j.requires_grad_(True)
        lz = F.l1_loss(images, fake_imgs2)
        # lz += -F.cosine_similarity(j.view(*list(l.shape)), l)
        lz += (j.view(*list(l.shape)) != l).float().mean()
        lz += self.tr(j.view(*list(l.shape)).float(), l.float(), torch.randint(0, 8192, l.shape, device=self.device).float())
        self.log("L1", lz, on_step=True, on_epoch=True, prog_bar=True)
        return lz


    def configure_optimizers(self):
        self.g_optim = optim.Adam(list(self.enc.parameters()), lr=G_LR, betas=ADAM_BETA)
        return [self.g_optim], []
        
    
    def validation_step(self, batch, batch_idx):
        pass
        
    
    def validation_epoch_end(self, outputs):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
        
    
    def test_epoch_end(self, outputs):
        pass
