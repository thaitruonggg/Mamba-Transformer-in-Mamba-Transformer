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
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math

def moex(x, swap_index, norm_type, epsilon=1e-5, positive_only=False):
    '''MoEx operation'''
    dtype = x.dtype
    x = x.float()

    # Determine if input is 1D sequence or 2D image
    if x.dim() == 3:  # [B, C, L] format
        B, C, L = x.shape
        H, W = 1, L  # Treat as 1D sequence
        is_1d = True
    else:  # [B, C, H, W] format
        B, C, H, W = x.shape
        is_1d = False

    # Reshape 1D to 2D if needed for consistent processing
    if is_1d:
        x = x.view(B, C, H, W)

    if norm_type == 'bn':
        norm_dims = [0, 2, 3]
    elif norm_type == 'in':
        norm_dims = [2, 3]
    elif norm_type == 'ln':
        norm_dims = [1, 2, 3]
    elif norm_type == 'pono':
        norm_dims = [1]
    elif norm_type.startswith('gn'):
        if norm_type.startswith('gn-d'):
            # gn-d4 means GN where each group has 4 dims
            G_dim = int(norm_type[4:])
            G = C // G_dim
        else:
            # gn4 means GN with 4 groups
            G = int(norm_type[2:])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2, 3, 4]
    elif norm_type.startswith('gpono'):
        if norm_type.startswith('gpono-d'):
            # gpono-d4 means GPONO where each group has 4 dims
            G_dim = int(norm_type[len('gpono-d'):])
            G = C // G_dim
        else:
            # gpono4 means GPONO with 4 groups
            G = int(norm_type[len('gpono'):])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2]
    else:
        raise NotImplementedError(f'norm_type={norm_type}')

    if positive_only:
        x_pos = F.relu(x)
        s1 = x_pos.sum(dim=norm_dims, keepdim=True)
        s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
        count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
        count[count == 0] = 1  # deal with 0/0
        mean = s1 / count
        var = s2 / count - mean.pow(2)
        std = var.add(epsilon).sqrt()
    else:
        mean = x.mean(dim=norm_dims, keepdim=True)
        std = x.var(dim=norm_dims, keepdim=True).add(epsilon).sqrt()

    swap_mean = mean[swap_index]
    swap_std = std[swap_index]

    # output = (x - mean) / std * swap_std + swap_mean
    # equivalent but more efficient
    scale = swap_std / std
    shift = swap_mean - mean * scale
    output = x * scale + shift

    # Return to original shape if input was 1D
    if is_1d:
        return output.view(B, C, L).to(dtype)
    else:
        return output.view(B, C, H, W).to(dtype)

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

    def forward(self, pixel_embed, patch_embed, mask=None):
        # Get original batch sizes
        b_pixel, n, _ = pixel_embed.shape
        b_patch, m, _ = patch_embed.shape
        h = self.heads

        # Check if batch sizes match
        if b_pixel != b_patch:
            # Reshape pixel_embed to match patch_embed's batch size
            # This assumes pixel_embed has been reshaped elsewhere in the code
            # and needs to be restored to the original batch size
            real_batch_size = b_patch
            pixels_per_patch = b_pixel // real_batch_size
            pixel_embed = pixel_embed.reshape(real_batch_size, pixels_per_patch * n, -1)

        q = self.to_q(pixel_embed)  # (b, n, inner_dim)
        k = self.to_k(patch_embed)  # (b, m, inner_dim)
        v = self.to_v(patch_embed)  # (b, m, inner_dim)

        #q, k, v = map(lambda t: rearrange(t, 'b seq (h d) -> (b h) seq d', h=h), (q, k, v))
        q = rearrange(q, 'b seq (h d) -> (b h) seq d', h=h)
        k = rearrange(k, 'b seq (h d) -> (b h) seq d', h=h)
        v = rearrange(v, 'b seq (h d) -> (b h) seq d', h=h)

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

# Implement Mambavision
class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Block(nn.Module):
    """ TNT Block with MambaVision Mixer
    """

    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        # First block: Norm → MambaVision Mixer → Add
        self.norm_in_mamba = norm_layer(in_dim)
        self.mamba_in = MambaVisionMixer(
            d_model=in_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # Second block: Norm → Attention → Add
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # Third block: Norm → MLP → Add
        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
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

        # Add cross-attention
        self.cross_attn = CrossAttention(query_dim=in_dim, context_dim=dim, heads=in_num_head)

    def forward(self, pixel_embed, patch_embed):
        # Inner transformer
        # First: Norm → MambaVision Mixer → Add
        pixel_embed = pixel_embed + self.drop_path(self.mamba_in(self.norm_in_mamba(pixel_embed)))

        # Second: Norm → Attention → Add
        x, _ = self.attn_in(self.norm_in(pixel_embed))
        pixel_embed = pixel_embed + self.drop_path(x)

        # Third: Norm → MLP → Add
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))

        # Outer transformer
        B, N, C = patch_embed.size()
        Nsqrt = int(math.sqrt(N))

        # Create a new tensor instead of modifying in-place
        new_patch_embed = patch_embed.clone()
        new_patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        patch_embed = new_patch_embed

        x, weights = self.attn_out(self.norm_out(patch_embed))
        patch_embed = patch_embed + self.drop_path(x)

        # Apply cross-attention (non-in-place)
        pixel_embed = self.cross_attn(pixel_embed, patch_embed)

        # Split and process without in-place operations
        cls_token = patch_embed[:, 0:1]
        patch_tokens = patch_embed[:, 1:]

        patch_tokens = patch_tokens.transpose(1, 2).view(B, C, Nsqrt, Nsqrt)
        patch_tokens = self.conv(patch_tokens).flatten(2).transpose(1, 2)

        # Concatenate back together
        patch_embed = torch.cat([cls_token, patch_tokens], dim=1)

        return pixel_embed, patch_embed, weights


class LocalViT_TNT(TNT):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=32, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=8):
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

    def forward_features(self, x, swap_index, moex_norm, moex_epsilon, moex_layer, moex_positive_only):
        attn_weights = []
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        # moex
        if swap_index is not None and moex_layer == 'stem':
            patch_embed = moex(patch_embed, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        for blk in self.blocks:
            pixel_embed, patch_embed, weights = blk(pixel_embed, patch_embed)
            attn_weights.append(weights)
        patch_embed = self.norm(patch_embed)
        return patch_embed[:, 0], attn_weights

    def forward(self, x, swap_index=None, moex_norm='pono', moex_epsilon=1e-5,
                moex_layer='stem', moex_positive_only=False, vis=False):
        x, attn_weights = self.forward_features(x, swap_index, moex_norm, moex_epsilon,
                                                moex_layer, moex_positive_only)
        x = self.head(x)
        if vis:
            return x, attn_weights
        else:
            return x


@register_model
def LNL_MoEx_Ti(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def LNL_MoEx_S(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model