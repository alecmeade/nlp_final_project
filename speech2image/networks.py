import math
import random
import torch
from collections  import OrderedDict
from functools    import partial
from torch import nn
from speechbrain.lobes.models.transformer.conformer import ConformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder2
from .layers import *


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], latent_dim=512):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        # self.concat = nn.Linear((channels[4] * 4 * 4) + latent_dim, channels[4] * 4 * 4)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, label=None):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        # if label is not None:
            # out = self.concat(torch.cat([out, label[0]], dim=1))
        out = self.final_linear(out)

        return out


class Encoder(nn.Module):
    def __init__(self, n_mels=40, latent_dim=256):
        super().__init__()
        self.enc = ConformerEncoder(num_layers=2, nhead=2, d_ffn=128, d_model=n_mels)
        self.linear = nn.Linear(n_mels, latent_dim)

    def forward(self, input, lens, r=True):
        x = input.permute(0, 2, 1)
        # masks = torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None]
        x = self.enc(x)[0]
        if r:
            x = self.linear(F.relu(x.mean(dim=1)))
        return x


class ESPNet2Encoder(nn.Module):
    def __init__(self, n_mels=40, latent_dim=256):
        super().__init__()
        self.enc = ConformerEncoder2(
            n_mels,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="conv2d",
            normalize_before=True,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31
        )
    
    def forward(self, input, lens):
        return self.enc(input, lens)


# Simple GAN
class SimpleGenerator(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        model = []
        model.append(nn.Conv2d(self.latent_size, 256, kernel_size=4, stride=1, padding=3, bias=False)) # Modified to k=8, p=7 for our image dimensions (i.e. 512x512)
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())

        model.append(nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Tanh())
        self.model = nn.Sequential(*model)


class SimpleDiscriminator(nn.Module):
    def __init__(self, label_size=512):
        super().__init__()
        self.label_size = label_size
        self.build_model(label_size//(4 * 4))

    def forward(self, x, l):
        x = self.model(x)
        k = torch.cat((x, l.reshape(x.shape[0], -1, 4, 4)), 1)
        return self.output(k)

    def build_model(self, s):
        model = []
        model.append(nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(MiniBatchAverageLayer())
        model.append(nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))

        output = [] # After the label concatenation
        output.append(nn.Conv2d(256 + s, 256, kernel_size=1, stride=1, padding=0, bias=False))
            
        output.append(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # model.append(nn.Sigmoid()) # Output probability (in [0, 1])
        self.model = nn.Sequential(*model)
        self.output = nn.Sequential(*output)

class SimplestGenerator(nn.Module):
    def __init__(self, img_size=32, latent_dim=512):
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class SimplestDiscriminator(nn.Module):
    def __init__(self, img_size=32, label_dim=512):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(out_filters, out_filters, 3, 1, 1), nn.LeakyReLU(0.2)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2 + label_dim, 1), nn.Sigmoid())

    def forward(self, img, label):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(torch.cat([out, label], dim=-1))

        return validity
