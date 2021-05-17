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
ENC_LR = 1e-5
ADAM_BETA = (0.0, 0.99)
CHANNEL_MULTIPLIER = 2


class Speech2ImageSC(pl.LightningModule):
    def __init__(self, img_size=256, latent=512, n_mlp=8, pretrained=None, audio_davenet=None, image_davenet=None):
        super().__init__()

        self.latent_size = latent
        self.audio_davenet = audio_davenet
        self.image_davenet = image_davenet
        self._init_networks(img_size, self.latent_size, n_mlp, pretrained)

    def _init_networks(self, img_size, latent, n_mlp, pretrained=None):
        self.enc = ESPnetEncoder.from_pretrained()
        self.enc.train()
        self.audio_davenet.eval()
        self.image_davenet.eval()
        self.G = Generator(img_size, latent, n_mlp, channel_multiplier=CHANNEL_MULTIPLIER)
        self.D = Discriminator(img_size, channel_multiplier=CHANNEL_MULTIPLIER)
        self.G_EMA = copy.deepcopy(self.G).eval()
    
    def del_networks(self):
        del self.G
        del self.D

    def forward(self, x=None, nframes=None):
        x = self.enc(x)
        z = x.mean(dim=1).view(1, x.shape[0], -1)
        z = torch.cat([z, torch.randn(1, x.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        return self.G_EMA(z, randomize_noise=False)[0]

    def training_step(self, batch, batch_idx):
        images, (audio, spec), nframes, apath = batch

        # Generate image
        x = self.enc(audio)
        z = x.mean(dim=1).view(1, audio.shape[0], -1)
        z = torch.cat([z, torch.randn(1, audio.shape[0], z.shape[-1], device=self.device)], dim=0).unbind(0)
        fake_imgs, _ = self.G_EMA(z, randomize_noise=False)

        image_output, image_dim = self._get_imagefeatures(fake_imgs)
        audio_output = self._get_audiofeatures(spec)
        _, img_outputH, img_outputW = image_dim
        heatmap = torch.bmm(audio_output.permute(0, 2, 1), image_output).squeeze()
        heatmap = heatmap.view(audio_output.size(0), audio_output.size(2), img_outputH, img_outputW)

        B = audio_output.size(0)
        N_t = audio_output.size(2)
        N_r = img_outputH
        N_c = img_outputW
        sisa = torch.sum(heatmap) / (B * N_t * N_r * N_c)
        misa = torch.sum(torch.amax(heatmap.view(B, N_t, N_r * N_c), dim=2))/(N_t)
        sima = torch.sum(torch.amax(heatmap, dim=2))/(B * N_r * N_c)
        return -(sisa + misa + sima)

    def _get_imagefeatures(self, img):
        image_feature_map = self.image_davenet(img)
        batch_dim = image_feature_map.size(0)
        emb_dim = image_feature_map.size(1)
        output_H = image_feature_map.size(2)
        output_W = image_feature_map.size(3)
        return image_feature_map.view(batch_dim, emb_dim, output_H * output_W), (emb_dim, output_H, output_W)

    def _get_audiofeatures(self, melspec):
        audio_output = self.audio_davenet(melspec)
        return audio_output

    def configure_optimizers(self):
        self.enc_optim = optim.Adam(self.enc.parameters(), lr=ENC_LR, betas=ADAM_BETA)
        return self.enc_optim
    
    def validation_step(self, batch, batch_idx):
        images, (audio, _), nframes, apath = batch
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
        images, (audio, _), nframes, apath = batch
        fake_imgs = self.forward(audio, nframes).cpu()
        return {"G_IMGs": fake_imgs, "I_AUDs": apath, "R_IMGs": images.cpu()}
        
    def test_epoch_end(self, outputs):
        return outputs
