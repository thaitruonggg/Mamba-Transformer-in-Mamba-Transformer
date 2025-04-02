from os.path import exists
from inspect import isfunction
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
from models.localvit import LocalityFeedForward
from models.tnt import Attention, TNT
import math


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_t_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module): # Pixel_embed
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, patch_embed, pixel_embed, mask=None):
        b, n, _ = patch_embed.shape  # Now using patch_embed's shape
        h = self.heads

        q = self.to_q(patch_embed)   # Now querying from patch_embed
        k = self.to_k(pixel_embed)   # Keys from pixel_embed
        v = self.to_v(pixel_embed)   # Values from pixel_embed

        q, k, v = map(lambda t: rearrange(t, 'b seq (h d) -> (b h) seq d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) seq d -> b seq (h d)', h=h)

        out = self.to_out(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)

        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, reduction=dim)

        # Add CrossAttention - switched the query_dim and context_dim
        self.cross_attn = CrossAttention(query_dim=dim, context_dim=in_dim, heads=in_num_head)

    def forward(self, pixel_embed, patch_embed):
        # inner transformer
        x, _ = self.attn_in(self.norm_in(pixel_embed))
        pixel_embed = pixel_embed + self.drop_path(x)
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))

        # outer transformer
        B, N, C = patch_embed.size()
        Nsqrt = int(math.sqrt(N))
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        x, weights = self.attn_out(self.norm_out(patch_embed))
        patch_embed = patch_embed + self.drop_path(x)

        # cross-attention - swapped the order of arguments
        # Now patch_embed provides the query and pixel_embed provides the key and value
        patch_embed = self.cross_attn(patch_embed, pixel_embed)

        cls_token, patch_embed = torch.split(patch_embed, [1, N - 1], dim=1)
        patch_embed = patch_embed.transpose(1, 2).view(B, C, Nsqrt, Nsqrt)
        patch_embed = self.conv(patch_embed).flatten(2).transpose(1, 2)
        patch_embed = torch.cat([cls_token, patch_embed], dim=1)

        return pixel_embed, patch_embed, weights


class LocalViT_TNT(TNT):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=32, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=8): #Old patch_size=16, first_stride=4
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, in_dim, depth,
                         num_heads, in_num_head, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                         drop_path_rate, norm_layer, first_stride)
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.apply(self._init_weights)


@register_model
def LNL_Ti(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def LNL_S(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model