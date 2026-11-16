"""Models.

All three experiments share one feature extractor, :class:`ResNet18Trunk`, so the downstream
classifier (trunk -> global average pool -> head) is *architecturally identical* for the
supervised baseline, the standard AE and the overcomplete sparse AE. The autoencoders differ only in
the width of a 1x1 latent projection on top of the trunk (and, through the loss, in sparsity /
denoising). The latent projection and the decoder are pre-training-only modules and are never
instantiated in the downstream classifier.

STL-10 adaptation of ResNet-18
------------------------------
ImageNet ResNet-18 uses a 7x7/2 conv + 3x3/2 max-pool stem (224 -> 56). On 96x96 inputs that would
leave a 3x3 map after layer4. We use a 3x3/2 conv stem without max-pool (96 -> 48), giving
48 -> 24 -> 12 -> 6 spatial resolution through layer1..layer4. Block structure, widths
(64/128/256/512) and depth (2-2-2-2 BasicBlocks) are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IMG_SIZE, INPUT_DIM, NUM_CLASSES


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18Trunk(nn.Module):
    """ResNet-18 convolutional trunk for 96x96 inputs. Output: (B, 512, 6, 6)."""

    out_channels = 512

    def __init__(self, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            blk = [BasicBlock(cin, w, stride)] + [BasicBlock(w, w, 1) for _ in range(n - 1)]
            layers.append(nn.Sequential(*blk))
            cin = w
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class Classifier(nn.Module):
    """Downstream model used by every experiment: trunk -> GAP (512-d) -> BN (no affine) -> Linear.

    The parameter-free BatchNorm1d standardises features before the linear layer (standard practice
    for linear probes); it is kept in the fine-tuning protocol too so all models share one architecture.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ResNet18Trunk()
        self.head = nn.Sequential(nn.BatchNorm1d(ResNet18Trunk.out_channels, affine=False),
                                  nn.Linear(ResNet18Trunk.out_channels, num_classes))

    def features(self, x):
        return F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with a ResNet-18 encoder and a ReLU latent code z (B, C_z, 6, 6).

    * Undercomplete AE : C_z = 128  ->  4,608 latent coefficients (0.17x the 27,648 input values)
    * Overcomplete AE  : C_z = 2048 -> 73,728 latent coefficients (2.67x the input)

    The decoder contains **no normalisation layers** and its first (1x1) layer has columns constrained
    to unit L2 norm (one column per latent channel, re-projected after every optimiser step). Without
    this, an L1 penalty on z can be driven to zero trivially by shrinking z and scaling up the decoder
    (or by a downstream BatchNorm undoing the scale), so the sparsity term would be meaningless.
    """

    def __init__(self, latent_channels: int, dec_widths=(256, 128, 64, 32), unit_norm_decoder: bool = True):
        super().__init__()
        self.latent_channels = latent_channels
        self.unit_norm_decoder = unit_norm_decoder
        self.encoder = ResNet18Trunk()
        self.to_latent = nn.Conv2d(ResNet18Trunk.out_channels, latent_channels, 1)
        self.from_latent = nn.Conv2d(latent_channels, ResNet18Trunk.out_channels, 1)
        layers, c = [nn.ReLU(inplace=True)], ResNet18Trunk.out_channels
        for w in dec_widths:  # 6 -> 12 -> 24 -> 48 -> 96
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, w, 3, 1, 1), nn.ReLU(inplace=True)]
            c = w
        layers.append(nn.Conv2d(c, 3, 3, 1, 1))
        self.decoder = nn.Sequential(*layers)
        self.renorm_decoder()

    def encode(self, x):
        return F.relu(self.to_latent(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.from_latent(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm_decoder(self):
        if self.unit_norm_decoder:
            w = self.from_latent.weight  # (512, C_z, 1, 1)
            w.div_(torch.linalg.vector_norm(w, dim=(0, 2, 3), keepdim=True).clamp_min(1e-8))

    def latent_shape(self) -> tuple[int, int, int]:
        s = IMG_SIZE // 16
        return (self.latent_channels, s, s)

    def latent_numel(self) -> int:
        c, h, w = self.latent_shape()
        return c * h * w

    def overcompleteness(self) -> float:
        return self.latent_numel() / INPUT_DIM

    def param_breakdown(self) -> dict:
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {
            "encoder_trunk": n(self.encoder),
            "latent_projection": n(self.to_latent),
            "encoder_total": n(self.encoder) + n(self.to_latent),
            "decoder_total": n(self.from_latent) + n(self.decoder),
            "total": n(self),
        }


def load_trunk_into(classifier: Classifier, encoder_ckpt: str) -> None:
    """Copy pre-trained ResNet-18 trunk weights (from scripts/pretrain.py) into a classifier.

    Only the trunk is transferred; the latent projection and decoder are discarded.
    """
    ck = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    classifier.encoder.load_state_dict(ck["encoder"], strict=True)  # strict: architectures must match
