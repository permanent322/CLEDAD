import argparse
import datetime
import os

import torch.nn.functional as F
import numpy as np
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from .diffEmbedding import DiffusionEmbedding


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """
    Initializes a 1D convolutional layer with Kaiming normal initialization.

    Parameters:
    - in_channels (int): Number of channels in the input signal.
    - out_channels (int): Number of channels produced by the convolution.
    - kernel_size (int): Size of the convolving kernel.

    Returns:
    - nn.Conv1d: A 1D convolutional layer with weights initialized.
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer





class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter=True
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x

class ResidualExtractor(nn.Module):
    def forward(self, x, trend, seasonality):
        # 残差部分 = 原始序列 - 趋势部分 - 周期部分
        residual = x - trend - seasonality
        return residual

# class ResidualProcessor(nn.Module):
#     def __init__(self, dim):
#         super(ResidualProcessor, self).__init__()
#         self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
#         self.act = nn.ReLU()
#
#     def forward(self, residual):
#         # 通过卷积层处理残差部分，然后通过ReLU激活函数
#         processed_residual = self.act(self.conv(residual))
#         return processed_residual

class ResidualProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualProcessor, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, residual):
        # 先通过1x1卷积层调整通道数
        residual = self.conv1x1(residual)
        # 然后通过卷积层处理残差部分，并通过ReLU激活函数
        processed_residual = self.act(self.conv(residual))
        return processed_residual


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim,mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ICB = True
        self.ASB = True

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # 残差部分提取和处理
        self.residual_extractor = ResidualExtractor()
        self.residual_processor = ResidualProcessor(in_channels=6, out_channels=6)

    def forward(self, x, diffusion_emb, mts_emb):

        # 提取趋势部分
        trend = self.drop_path(self.icb(self.norm2(x)))

        # 提取周期性部分
        seasonality = self.drop_path(self.asb(self.norm1(x)))

        # 提取残差部分
        residual = self.residual_extractor(x, trend, seasonality)

        # 处理残差部分
        residual_processed = self.residual_processor(residual)

        # Check if both ASB and ICB are true
        if self.ICB and self.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        # print("x.shape", x.shape)
        # print("residual_processed.shape", residual_processed.shape)
        x = x + residual_processed

        return x


class TSLANet_with_residual(nn.Module):

    def __init__(self, config):
        super(TSLANet_with_residual, self).__init__()


        self.patch_size= 16
        self.stride = self.patch_size
        num_patches = int((config["seq_len"] - self.patch_size) / self.stride + 1)
        self.channels = config["channels"]
        self.input_projection = Conv1d_with_init(1, self.channels, 1)
        # diffusion time step embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.diffusion_projection = nn.Linear(config["diffusion_embedding_dim"], self.channels)
        self.cond_projection = Conv1d_with_init(config["mts_emb_dim"], self.channels, 1)
        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, 128)
        self.mid_projection = Conv1d_with_init(self.channels, 1, 1)
        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]  # stochastic depth decay rule



        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=128, drop=0.5, drop_path=dpr[i])
            for i in range(5)]
        )

        # Parameters/Embeddings
        # self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)
        self.out_layer = nn.Linear(128 * num_patches, config["seq_len"])



    # def pretrain(self, x_in):
    #     # x_in = replace_nan_with_zero(x_in)
    #     device = next(self.parameters()).device
    #     x = x_in.to(device)
    #     x = rearrange(x_in, 'b l m -> b m l')
    #     x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    #     x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')
    #
    #     # xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
    #     xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=0.1)
    #     self.mask = self.mask.bool()  # mask: [bs x num_patch]
    #     xb_mask = self.input_layer(xb_mask)
    #
    #     for tsla_blk in self.tsla_blocks:
    #         xb_mask = tsla_blk(xb_mask)
    #
    #     return xb_mask, self.input_layer(x_patched)


    def forward(self, x, mts_emb, t): # x: [bs x 1 x dim x seq_len]
        device = next(self.parameters()).device

        x = x.to(device)
        x = x.float()
        B, inputdim, K, L = x.shape

        original_x = x.reshape(B, K, L)

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)  ## First Convolution before fedding the data to the
        x = F.relu(x)  ## residual block


        x = x.reshape(B, self.channels, L * K)
        diffusion_emb = self.diffusion_embedding(t)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        x = x + diffusion_emb

        _, mts_emb_dim, _, _ = mts_emb.shape
        mts_emb = mts_emb.reshape(B, mts_emb_dim, K * L)  # B, C,
        mts_emb = self.cond_projection(mts_emb)  # (B,2*channel,K*L)
        x = x - mts_emb
        x = self.mid_projection(x)
        x = x.reshape(B, L, K)

        B, L, M = x.shape


        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')  # （, , patch_size）
        x = self.input_layer(x)   # (, , emb_dim)

        # x = x + pos_encoding
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x, diffusion_emb, mts_emb)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs.permute(0, 2, 1)  # (B, K, L)

        predicted_noise = original_x - outputs

        return predicted_noise, outputs




if __name__ == '__main__':
    data = torch.randn(32, 1, 25, 128)
    mts_emb = torch.randn(32, 33, 25,128)
    t = torch.randn(32, ).long()
    diffmodel = TSLANet_with_residual(config={"num_steps": 128, "diffusion_embedding_dim": 128, "mts_emb_dim": 33},channels=64)
    output = diffmodel(data, mts_emb, t)
    print(output.shape)