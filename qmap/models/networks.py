"""Shared-encoder attention-fusion U-Net, with or without acquisition-conditioned AdaIN.

Module attribute names (and hence state_dict keys) and the order in which modules are created
are those of the original implementation: the published checkpoints load strictly, and a
fixed seed yields the same initial weights. Class names must not contain 'Conv', 'Linear' or
'BatchNorm2d', because `init_weights` selects modules by class name.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

# Conditioning vector per contrast (paper Eq. 9):
# [is_T1w, is_T2w, is_FLAIR, ln TR, ln TE, sin FA, B1rms, has_TI, ln TI]
COND_DIM = 9


def instance_norm(channels):
    return nn.InstanceNorm2d(channels, affine=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = instance_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = instance_norm(out_channels)
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_connection(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels)
        self.strided_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.residual_block(x)
        return x, self.strided_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            instance_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            instance_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        return self.conv(x)


class AttentionFusion(nn.Module):
    """Permutation-invariant fusion of the per-contrast feature maps (shared MLP + softmax)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

    def forward(self, feats):
        summaries = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)  # B x M x C
        w = torch.softmax(self.mlp(summaries), dim=1)                        # softmax over contrasts
        out = torch.zeros_like(feats[0])
        for m, f in enumerate(feats):
            out = out + w[:, m].unsqueeze(-1).unsqueeze(-1) * f
        return out


class SharedAttnUNet(nn.Module):
    """U-Net with a shared encoder per contrast and attention fusion ("No AdaIN" model)."""
    def __init__(self, input_channels=1, output_channels=4, feature_sizes=(64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        in_ch = input_channels
        for fs in feature_sizes:
            self.enc_blocks.append(EncoderBlock(in_ch, fs))
            in_ch = fs

        bottleneck_ch = feature_sizes[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_sizes[-1], bottleneck_ch, 3, padding=1),
            instance_norm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            instance_norm(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        self.skip_fusers = nn.ModuleList([AttentionFusion(fs) for fs in feature_sizes])
        self.bottleneck_fuser = AttentionFusion(bottleneck_ch)

        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for fs in reversed(feature_sizes):
            self.dec_blocks.append(DecoderBlock(prev_ch, fs))
            prev_ch = fs

        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_sizes[0], output_channels, 1),
            nn.Sigmoid(),
        )

    def _encode_one(self, x):
        skips = []
        for enc in self.enc_blocks:
            s, x = enc(x)
            skips.append(s)
        return skips, self.bottleneck(x)

    def forward(self, inputs):
        all_skips = [[] for _ in self.enc_blocks]
        all_bottlenecks = []
        for x in inputs:
            skips, bott = self._encode_one(x)
            for i, s in enumerate(skips):
                all_skips[i].append(s)
            all_bottlenecks.append(bott)

        fused_skips = [f(s) for f, s in zip(self.skip_fusers, all_skips)]
        x = self.bottleneck_fuser(all_bottlenecks)
        for dec, skip in zip(self.dec_blocks, reversed(fused_skips)):
            x = dec(x, skip)
        return self.output_layer(x)


class AdaIN(nn.Module):
    """Acquisition-conditioned adaptive instance normalization (paper Eqs. 10-11).

    gamma = W_g w + b_g + 1, beta = W_b w + b_b. The projection is zero-initialised here, but
    `init_weights` later re-initialises every Linear layer (Kaiming); the published models
    were trained that way, only the +1 offset on gamma remains.
    """
    def __init__(self, num_features: int, condition_length: int):
        super().__init__()
        self.num_features = num_features
        self.inst_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.linear = nn.Linear(condition_length, num_features * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_t, condition):
        normed = self.inst_norm(input_t)
        gamma_and_beta = self.linear(condition)
        gamma = gamma_and_beta[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_and_beta[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)
        gamma = gamma + 1
        return gamma * normed + beta


class ResidualBlockAdaIN(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cin1 = AdaIN(out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cin2 = AdaIN(out_channels, cond_dim)
        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x, c):
        residual = self.residual_connection(x)
        x = F.relu(self.cin1(self.conv1(x), c), inplace=True)
        x = self.cin2(self.conv2(x), c)
        x += residual
        return F.relu(x, inplace=True)


class EncoderBlockAdaIN(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.residual_block = ResidualBlockAdaIN(in_channels, out_channels, cond_dim)
        self.strided_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.residual_block(x, c)
        return x, self.strided_conv(x)


class AdaINSharedAttnUNet(nn.Module):
    """Shared-encoder attention U-Net whose encoder and bottleneck use AdaIN (proposed model)."""
    def __init__(self, input_channels=1, output_channels=4, feature_sizes=(64, 128, 256),
                 cond_input_dim=COND_DIM, mlp_layers=2, embed_dim=128):
        super().__init__()
        # Conditioning MLP with ELU activations; hidden widths halve going backwards from the
        # embedding size (2 layers: 9 -> 64 -> 128).
        if mlp_layers == 1:
            mlp = [nn.Linear(cond_input_dim, embed_dim)]
        else:
            hidden_dims = [max(embed_dim // (2 ** i), 8) for i in range(mlp_layers - 1, 0, -1)]
            mlp, curr_dim = [], cond_input_dim
            for h_dim in hidden_dims:
                mlp += [nn.Linear(curr_dim, h_dim), nn.ELU(inplace=True)]
                curr_dim = h_dim
            mlp.append(nn.Linear(curr_dim, embed_dim))
        self.cond_mlp = nn.Sequential(*mlp)

        self.enc_blocks = nn.ModuleList()
        in_ch = input_channels
        for fs in feature_sizes:
            self.enc_blocks.append(EncoderBlockAdaIN(in_ch, fs, embed_dim))
            in_ch = fs

        bottleneck_ch = feature_sizes[-1] * 2
        self.bottleneck_conv1 = nn.Conv2d(feature_sizes[-1], bottleneck_ch, 3, padding=1)
        self.bottleneck_cin1 = AdaIN(bottleneck_ch, embed_dim)
        self.bottleneck_conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1)
        self.bottleneck_cin2 = AdaIN(bottleneck_ch, embed_dim)

        self.skip_fusers = nn.ModuleList([AttentionFusion(fs) for fs in feature_sizes])
        self.bottleneck_fuser = AttentionFusion(bottleneck_ch)

        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for fs in reversed(feature_sizes):
            self.dec_blocks.append(DecoderBlock(prev_ch, fs))
            prev_ch = fs

        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_sizes[0], output_channels, 1),
            nn.Sigmoid()
        )

    def _encode_one(self, x, c):
        skips = []
        c_emb = self.cond_mlp(c)
        for enc in self.enc_blocks:
            s, x = enc(x, c_emb)
            skips.append(s)
        x = self.bottleneck_conv1(x)
        x = F.relu(self.bottleneck_cin1(x, c_emb), inplace=True)
        x = self.bottleneck_conv2(x)
        x = F.relu(self.bottleneck_cin2(x, c_emb), inplace=True)
        return skips, x

    def forward(self, inputs, conditions):
        all_skips = [[] for _ in self.enc_blocks]
        all_bottlenecks = []
        for x, c in zip(inputs, conditions):
            skips, bott = self._encode_one(x, c)
            for i, s in enumerate(skips):
                all_skips[i].append(s)
            all_bottlenecks.append(bott)

        fused_skips = [f(s) for f, s in zip(self.skip_fusers, all_skips)]
        x = self.bottleneck_fuser(all_bottlenecks)
        for dec, skip in zip(self.dec_blocks, reversed(fused_skips)):
            x = dec(x, skip)
        return self.output_layer(x)


def init_weights(net):
    """Kaiming-normal weights and zero biases for every Conv/ConvTranspose/Linear layer."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def build_generator(model_cfg, device, output_channels=4):
    """Construct the generator on `device` and initialise it.

    The weights are initialised after moving the network to the device, so on a GPU they are
    drawn from the CUDA generator (as for the published models).
    """
    feature_sizes = list(model_cfg.feature_channels)
    if model_cfg.arch == 'adain':
        net = AdaINSharedAttnUNet(output_channels=output_channels, feature_sizes=feature_sizes,
                                  mlp_layers=model_cfg.adain.mlp_layers,
                                  embed_dim=model_cfg.adain.embed_dim)
    elif model_cfg.arch == 'no_adain':
        net = SharedAttnUNet(output_channels=output_channels, feature_sizes=feature_sizes)
    else:
        raise ValueError(f"Unknown model.arch '{model_cfg.arch}' (expected 'adain' or 'no_adain')")
    net.to(device)
    init_weights(net)
    return net
