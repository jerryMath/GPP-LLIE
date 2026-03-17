# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
from einops import rearrange
import numbers
import torch.nn.functional as F
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        out = out.to(torch.float32)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = torch.cat((o1, o2), dim=1)

        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                     self.dilation, self.groups, self.deformable_groups)


class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.dcn = DCNv2Pack(in_channel, in_channel, 3, padding=1, deformable_groups=4)

    def forward(self, x_vq):
        x_residual = self.offset(x_vq)

        feat_after_warp = self.dcn(x_vq, x_residual)

        return feat_after_warp


class OverlapPatchEmbed(nn.Module):
    """
    a tiny layer that turns an image (or any feature map) into an embedding map
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    The forward process gradually adds noise to the image across discrete timesteps 𝑡 = 1,…,𝑇
    The reverse model must know “where in the process” it is, because the denoising strategy at
    early steps (very noisy input) is different from later steps (almost clean input).Without
    embedding t, the network would see the same noisy image but not know how much noise it’s
    supposed to remove.

    If t is scaler:
    1. Get “washed out” among thousands of other features (especially in deep conv/transformer layers).
    2. Be too limited to represent complex, nonlinear changes across timesteps.
    Neural networks work better when the conditioning signal is high-dimensional and structured.
    The sinusoidal mapping turns t into a vector with multiple frequencies,
    giving the model a smooth, continuous, and unique representation of every timestep.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
## Better than plain FFN w.r.t sharper details
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # This is the key part: have info of local neighbours, critical for reconstructing fine
        # textures
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, dim, H, W) -> (B, 2h, H, W)
        x = self.project_in(x)
        # x1: (B, h, H, W)
        # x2: (B, h, H, W)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x: (B, h, H, W)
        # a * b: dot product
        """
        The split into two halves followed by GELU(x1) ⊙ x2 lets one branch learn features 
        while the other learns how much to pass, per pixel and per channel. This selectively
         amplifies structure and suppresses artifacts/noise—useful for SR edges, lattices, 
         and repeated textures the paper showcases. 
        """
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads  #
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # scalers

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # after [B, head, C_per_head, H*W]
        # head: how many heads we split into.
        # C_per_head = C / head: per-head channel depth.
        # H*W: number of “tokens” (each spatial position becomes one token).
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Transposed-attention of size c*c instead of hw*hw, less computation burden
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


## Multi-DConv Head Transposed Cross-Attention
class Cross_attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        #The 1×1 Conv learns K and V from the local features, and the 3×3
        # depthwise Conv adds local spatial structure so that
        # channel-attention has neighborhood awareness
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # depthwise conv, add spatial info
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, q_map):
        b, c, h, w = x.shape
        # print(f"=== b, c, h, w: {b, c, h, w}")
        # q_map from VLMs
        kv = self.kv_dwconv(self.kv(q_map))
        # print(f"=== kv: {kv.shape}")
        k, v = kv.chunk(2, dim=1)
        # print(f"=== k: {k.shape}")
        # print(f"=== v: {v.shape}")
        # [1, 256, 100, 150] -> [1, 8, 32, 15000], num heads is 8
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # print(f"=== k: {k.shape}")
        # print(f"=== v: {v.shape}")

        q = self.q_dwconv(self.q(x))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # weighted sum of all the channels of each pixel
        # for each head, each channel is replaced by a weighted sum of all channels
        """
        It does not mix spatial positions (no spatial attention) 
        It does mix channels per head at each position based on spatial similarity patterns.
        Channel-attention lets the model adjust how each pixel uses different feature channels 
        (texture, illumination, contrast, sharpness) without mixing spatial locations, 
        making it ideal for low-light enhancement.
        """
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


def modulate(x, shift, scale):
    """
    applies a per-channel scale and shift to a feature map,
    broadcasting across spatial dimensions
    """
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim * 2, LayerNorm_type)
        self.swarp = WarpBlock(dim * 2)
        self.attn = Attention(dim * 2, num_heads, bias)
        self.norm2 = LayerNorm(dim * 2, LayerNorm_type)
        self.ffn = FeedForward(dim * 2, ffn_expansion_factor, bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 12 * dim, bias=True)
        )

        self.map_conv = nn.Conv2d(3, dim * 2, kernel_size=3, stride=4, padding=1, bias=bias)
        self.map_norm = LayerNorm(dim * 2, LayerNorm_type)

        self.cross_norm = LayerNorm(dim * 2, LayerNorm_type)
        self.cross_attn = Cross_attention(dim * 2, num_heads, bias)

    def forward(self, x, y, t):

        # q_map = self.map_conv(q_map)
        # q_map = self.map_norm(q_map)

        # gate: DiT uses this to turn on/off parts of the block depending on timestep.
        """
        y = x + gate_msa * MSA( modulated_LN(x) )
        y = y + gate_mlp * MLP( modulated_LN(y) )
        gate only makes sense when you have a residual connection:
        out=x+gate⋅F(x)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        dim = x.shape[1]
        # print(f"=== x 1: {x.shape}")
        # print(f"=== y 1: {y.shape}")
        x = torch.cat([x, y], 1)
        # print(f"=== x 2: {x.shape}")
        x_ = self.swarp(x)

        x_ = modulate(self.norm1(x_), shift_msa, scale_msa) #GPP-LN: modulate
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.attn(x_)
        # print(f"=== x 3: {x.shape}")

        # 做cross_attention
        # x = x + self.cross_attn(self.cross_norm(x), q_map)
        # print(f"=== x 4: {x.shape}")

        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # print(f"=== x 5: {x.shape}")
        x = x[:, :dim, :, :]
        # print(f"=== x final: {x.shape}")
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, embed_dim, out_channel, LayerNorm_type):
        super().__init__()
        self.norm_final = LayerNorm(embed_dim, LayerNorm_type)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

        self.final_conv = nn.Conv2d(embed_dim, out_channel, 3, 1, 1)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.final_conv(x)
        return x


class DiT_incontext_revise(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            in_channels=3,
            embed_dim=128,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            learn_sigma=True,
            LayerNorm_type='WithBias',
            bias=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.x_embedder = OverlapPatchEmbed(in_c=in_channels, embed_dim=embed_dim, bias=bias)
        # Effect: Each timestep t gets a rich, smooth, continuous embedding, so the
        # model can condition on where it is in the diffusion process.
        self.t_embedder = TimestepEmbedder(embed_dim)  # vector，走scheme 1
        self.y_embedder = OverlapPatchEmbed(in_c=in_channels, embed_dim=embed_dim, bias=bias)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_expansion_factor=mlp_ratio, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(embed_dim, self.out_channels, LayerNorm_type)

    # def forward(self, x, t, y, vis, q_map):
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        B, _, h, w = x.shape
        # print(f"===x raw: {x.shape}")
        x = self.x_embedder(x)
        # print(f"===x embedded: {x.shape}")
        y = self.y_embedder(y)
        t = self.t_embedder(t)
        # remove local and global priors
        # t = t + torch.unsqueeze(vis, dim=-1).to(torch.float32)

        for j, block in enumerate(self.blocks):
            x = block(x, y, t)

        x = self.final_layer(x, t)

        return x
