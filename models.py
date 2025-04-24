#%%
import math
from functools import partial
import matplotlib.pyplot as plt
from packaging import version
import numpy as np
import torch.optim as optim
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from collections import namedtuple
# helpers functions

def odeSol(x0, ATb, sigma, model, nsteps=100):
    traj = torch.zeros(nsteps+1 , x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], device=x0.device)
    traj[0,:,:, :, :] = x0
    t = torch.zeros(x0.shape[0], device=x0.device)
    
    with torch.no_grad():
        h = 1/nsteps
        for i in range(nsteps):
            xt = traj[i,:,:, :, :]
            k1 = model(xt, t, ATb, sigma)
            k2 = model(xt+h*k1/2, t+h/2, ATb, sigma)
            k3 = model(xt+h*k2/2, t+h/2, ATb, sigma)
            k4 = model(xt+h*k3, t+h, ATb, sigma)
            
            traj[i+1, :, :, :, :] = traj[i, :, :, :, :] + h/6*(k1 + 2*k2 + 2*k3 + k4)
            t = t+h
            
    return traj  


def Id(nc):
    conv = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, groups=nc, bias=False)

    # Initialize the filters to the identity
    with torch.no_grad():
        for i in range(nc):
            conv.weight[i, 0, :, :] = torch.zeros((3, 3))
            conv.weight[i, 0, 1, 1] = 1.0
    return conv


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# small helper modules
class Upsample(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.conv = nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

class Block(Module):
    """ Basic block with a 3x3 convolution, RMSNorm, SiLU activation, and optional dropout."""
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # self.norm = RMSNorm(dim_out)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            # Scaling and shifting the input for the activation function
            # This can be based on a MLP output from time embedding, allowing for
            # time variant activations, scale is adjusted to be centered at identity
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class RMSNorm(Module):
    """
    Custom module for RMS normalization across C dimension.
    
    Normalize across C for vector of length 1
    Rescale weightings across C by a learnable parameter g and 1/sqrt(dim(C))
    
    Input: N x C x H x W x D
    Output: N x C x H x W x D
    """    
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

class UnetBlock2D(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(UnetBlock2D, self).__init__()
        self.layerNorm1 = nn.LayerNorm(shape)
        self.layerNorm2 = nn.LayerNorm([out_c, shape[1],shape[2]])
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        
        self.activation = nn.SiLU() 
        
    def forward(self, x):
        try:
            out = self.layerNorm1(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("NaN or Inf after layerNorm1")
                raise ValueError

            out = self.conv1(out)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.activation(out)

            out = self.conv2(out)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.activation(out)

            out = self.layerNorm2(out)
            out = self.conv3(out)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.activation(out)

            return out
        except Exception as e:
            print("NaN or Inf caught in UnetBlock2D!")
            print(f"x stats: min={x.min().item():.2e}, max={x.max().item():.2e}, mean={x.mean().item():.2e}")
            raise e


class resnetBlock(nn.Module):
    def __init__(self, dims, in_c, out_c, levels=5):
        super(resnetBlock, self).__init__()
        
        self.Open   = UnetBlock2D((in_c, dims[0], dims[1]), in_c, out_c)
        self.Blocks = nn.ModuleList()
        for i in range(levels):
            bi = UnetBlock2D((out_c, dims[0], dims[1]), out_c, out_c)
            self.Blocks.append(bi)
    
    def forward(self, x):
        x = self.Open(x)
        for i in range(len(self.Blocks)):
            x = x + self.Blocks[i](x)
        return x



class Downsample(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.conv = nn.Conv2d(dim, default(dim_out, dim), 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

#%%
# UNet implementation with data and noise embedding
class UNetFMG_DE_NE(nn.Module):
    def __init__(self, arch=[3, 16, 32, 64, 128], dims=torch.tensor([64,64]), time_emb_dim=256, noise_emb_dim=256):
        super(UNetFMG_DE_NE, self).__init__()
        
        self.time_embed = nn.Linear(1, time_emb_dim)
        self.noise_embed = nn.Linear(1, noise_emb_dim)
        
        self.DBlocks = nn.ModuleList()
        self.DTE = nn.ModuleList()
        self.DDE = nn.ModuleList()
        self.DNE = nn.ModuleList()

        for i in range(len(arch)-1):
            te = self._make_te(time_emb_dim, arch[i])
            ne = self._make_te(noise_emb_dim, arch[i])
            de = self._make_de(arch[0], arch[i])
            
            blk = resnetBlock((dims[0], dims[1]), arch[i], arch[i+1]) 
            
            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DDE.append(de) 
            self.DNE.append(ne)
            dims = dims//2
        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.ne_mid = self._make_te(noise_emb_dim, arch[-1])
        # bottlneck data embedding # Newly added!!!!!!!!!!!!!!!!!!
        self.de_mid = self._make_de(arch[0], arch[-1])
        
        self.blk_mid = resnetBlock((dims[0], dims[1]), arch[-1], arch[-1])
        
        self.UBlocks = nn.ModuleList()
        self.UTE = nn.ModuleList()
        self.UDE = nn.ModuleList()
        self.UNE = nn.ModuleList()
        self.Smth = nn.ModuleList()

        for i in np.flip(range(len(arch)-1)):
            dims = dims*2
            teu = self._make_te(time_emb_dim, arch[i+1])
            neu = self._make_te(noise_emb_dim, arch[i+1])
            de = self._make_de(arch[0], arch[i+1])
            
            blku = resnetBlock((dims[0], dims[1]), arch[i+1], arch[i])
            
            self.Smth.append(Id(arch[i]))
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UNE.append(neu)
            self.UDE.append(de)

        self.h = nn.Parameter(torch.ones(len(arch)))
        
    def Coarsen(self, x):
        return F.interpolate(x, scale_factor=0.5)
    
    def Refine(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')
    
    
    def forward(self, x, t, ATb, noise):
        t = self.time_embed(t.unsqueeze(1))
        noise = self.noise_embed(noise.unsqueeze(1))
        n = len(x)

        X = [x]
        for i in range(len(self.DBlocks)):

            scale = x.shape[-1]/ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            de = self.DDE[i](ATbI)
            te = self.DTE[i](t).reshape(n, -1, 1, 1)
            ne = self.DNE[i](t).reshape(n, -1, 1, 1) * ATbI
            
            x  = self.DBlocks[i](x + te + de + ne)
            x  = self.Coarsen(x)
            
            X.append(x)
            
        te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        ne_mid = self.ne_mid(noise).reshape(n, -1, 1, 1) 
        # I will try to add data_emb even at the bottleneck
        # previous paper didn't have data_emb in the bottleneck
        # dx = self.blk_mid(x + te_mid + ne_mid) 
        scale = X[-1][-1].shape[-1]/ATb.shape[-1]
        ATbI = F.interpolate(ATb, scale_factor=scale)
        de_mid = self.de_mid(ATbI) 
        dx = self.blk_mid(x + te_mid + de_mid + ne_mid) 
        x = X[-1] + dx 

        cnt = -1
        for i in range(len(self.DBlocks)):
            cnt = cnt-1
            x = self.Refine(x)
            scale = x.shape[-1]/ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            te = self.UTE[i](t).reshape(n, -1, 1, 1)
            ne = self.UNE[i](noise).reshape(n, -1, 1, 1) * ATbI
            
            de = self.UDE[i](ATbI)

            x  = self.Smth[i](X[cnt]) + self.h[i]*self.UBlocks[i](x + te + de + ne)
            
        return x

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def _make_de(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )


# %%
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Ensure at least 1 output channel for query, key, and value
        self.query = nn.Conv2d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.key = nn.Conv2d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Project inputs to query, key, and value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Compute attention scores
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Apply gamma and residual connection
        out = self.gamma * out + x
        return out


# %%
class UNetFMGAttention_DE_NE(nn.Module):
    def __init__(self, arch=[3, 16, 32, 64, 128], dims=torch.tensor([64,64]), time_emb_dim=256, noise_emb_dim=256):
        super(UNetFMGAttention_DE_NE, self).__init__()

        self.time_embed = nn.Linear(1, time_emb_dim)
        self.noise_embed = nn.Linear(1, noise_emb_dim)

        self.DBlocks = nn.ModuleList()
        self.DTE = nn.ModuleList()
        self.DDE = nn.ModuleList()
        self.DNE = nn.ModuleList()
        self.DAttention = nn.ModuleList()

        self.attn_enc_idx = len(arch) - 2  # Last encoder block index

        for i in range(len(arch) - 1):
            te = self._make_te(time_emb_dim, arch[i])
            ne = self._make_te(noise_emb_dim, arch[i])
            de = self._make_de(arch[0], arch[i])

            blk = resnetBlock((dims[0], dims[1]), arch[i], arch[i + 1])

            if i == self.attn_enc_idx:
                attention = SelfAttention(arch[i + 1])
            else:
                attention = nn.Identity()

            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DDE.append(de)
            self.DNE.append(ne)
            self.DAttention.append(attention)

            dims = dims // 2

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.ne_mid = self._make_te(noise_emb_dim, arch[-1])
        self.de_mid = self._make_de(arch[0], arch[-1])

        self.blk_mid = resnetBlock((dims[0], dims[1]), arch[-1], arch[-1])
        self.attention_mid = SelfAttention(arch[-1])  # Apply attention at bottleneck

        self.UBlocks = nn.ModuleList()
        self.UTE = nn.ModuleList()
        self.UDE = nn.ModuleList()
        self.UNE = nn.ModuleList()
        self.Smth = nn.ModuleList()
        self.UAttention = nn.ModuleList()

        self.attn_dec_idx = 0  # First decoder block (immediately after bottleneck)

        for i in np.flip(range(len(arch) - 1)):
            dims = dims * 2
            teu = self._make_te(time_emb_dim, arch[i + 1])
            neu = self._make_te(noise_emb_dim, arch[i + 1])
            de = self._make_de(arch[0], arch[i + 1])

            blku = resnetBlock((dims[0], dims[1]), arch[i + 1], arch[i])

            if i == self.attn_dec_idx:
                attention = SelfAttention(arch[i])
            else:
                attention = nn.Identity()

            self.Smth.append(Id(arch[i]))
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UNE.append(neu)
            self.UDE.append(de)
            self.UAttention.append(attention)

        self.h = nn.Parameter(torch.ones(len(arch)))

    def Coarsen(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

    def Refine(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t, ATb, noise):
        t = self.time_embed(t.unsqueeze(1))
        noise = self.noise_embed(noise.unsqueeze(1))
        n = len(x)

        X = [x]
        for i in range(len(self.DBlocks)):
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale, mode='bilinear', align_corners=True)
            de = self.DDE[i](ATbI)
            ATbI_embed = self.DDE[i](ATbI)
            ne = self.DNE[i](t).reshape(n, -1, 1, 1) * ATbI_embed
            te = self.DTE[i](t).reshape(n, -1, 1, 1)

            x = self.DBlocks[i](x + te + de + ne)
            x = self.DAttention[i](x)
            x = self.Coarsen(x)
            X.append(x)

        te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        ne_mid = self.ne_mid(noise).reshape(n, -1, 1, 1)
        scale = X[-1].shape[-1] / ATb.shape[-1]
        ATbI = F.interpolate(ATb, scale_factor=scale, mode='bilinear', align_corners=True)
        de_mid = self.de_mid(ATbI)
        dx = self.blk_mid(x + te_mid + de_mid + ne_mid)
        dx = self.attention_mid(dx)
        dx = torch.clamp(dx, -10, 10)
        x = X[-1] + dx

        cnt = -1
        for i in range(len(self.DBlocks)):
            cnt = cnt - 1
            x = self.Refine(x)
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale, mode='bilinear', align_corners=True)
            ATbI_embed = self.UDE[i](ATbI)
            ne = self.UNE[i](noise).reshape(n, -1, 1, 1) * ATbI_embed
            te = self.UTE[i](t).reshape(n, -1, 1, 1)

            combined = x + te + ATbI_embed + ne
            combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)  # handles NaN or inf
            combined = torch.clamp(combined, -10.0, 10.0)  # optional soft clipping
            X[cnt] = torch.clamp(X[cnt], -10, 10)
            x_check = x + te + ATbI_embed + ne
            # if torch.isnan(x_check).any():
            #     pass
            #     # print(f"[Decoder Block {i}] NaN detected in combined input")
            #     # print(f"x: {torch.isnan(x).any()}, te: {torch.isnan(te).any()}, ATbI_embed: {torch.isnan(ATbI_embed).any()}, ne: {torch.isnan(ne).any()}")
            #     # print(f"x max: {x.max().item()}, te max: {te.max().item()}, de max: {ATbI_embed.max().item()}, ne max: {ne.max().item()}")
            
            # if not torch.isfinite(self.h[i]):
            #     pass
            #     print(f"self.h[{i}] exploded: {self.h[i].item()}")
            #     self.h.data[i] = torch.tensor(1.0, device=self.h.device)
            # print(f"[Decoder Block {i}] combined min={combined.min().item():.2e}, max={combined.max().item():.2e}, mean={combined.mean().item():.2e}")
            x = self.Smth[i](X[cnt]) + self.h[i] * self.UBlocks[i](combined)
            # x = self.Smth[i](X[cnt]) + self.h[i] * self.UBlocks[i](x + te + ATbI_embed + ne)
            x = self.UAttention[i](x)

        return x

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def _make_de(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )
