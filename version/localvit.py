"""
Author: Omid Nejati
Email: omid_nejaty@alumni.iust.ac.ir

Introducing locality mechanism to "DeiT: Data-efficient Image Transformers".
"""
import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath
from timm.models.registry import register_model


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# New 2: Dynamic Kernel Size
class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False, kernel_size=3):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        self.kernel_size = kernel_size # New
        hidden_dim = int(in_dim * expand_ratio)

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

# New 3: Replace Attention with WindowAttention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Assumes square windows (e.g., 7)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# New 3: Adjust Block for WindowAttention
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act='hs+se', reduction=4, wo_dp_conv=False, dp_first=False,
                 window_size=7, kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act, reduction, wo_dp_conv, dp_first, kernel_size)
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        patch_size = int(math.sqrt(N - 1))  # Exclude CLS token
        assert patch_size % self.window_size == 0, "Patch size must be divisible by window size"

        # Split CLS token
        cls_token, x = torch.split(x, [1, N - 1], dim=1)  # (B, 1, C), (B, N-1, C)

        # Reshape to spatial grid
        x = x.view(B, patch_size, patch_size, C)

        # Cyclic shift (for odd layers, shift by window_size//2)
        shift_size = self.window_size // 2 if self.training else 0  # Shift only during training
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Partition into windows
        num_windows = (patch_size // self.window_size) ** 2
        x = x.view(B, patch_size // self.window_size, self.window_size,
                   patch_size // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)  # B*nW, Wh*Ww, C

        # Create attention mask for shifted windows
        if shift_size > 0:
            attn_mask = self.create_shifted_mask(patch_size, self.window_size, shift_size)
        else:
            attn_mask = None

        # Apply window attention
        x = self.attn(x, attn_mask)
        x = x.view(B, patch_size // self.window_size, patch_size // self.window_size,
                   self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, patch_size, patch_size, C)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        # Flatten back to token sequence and reattach CLS token
        x = x.view(B, -1, C)
        x = torch.cat([cls_token, x], dim=1)

        # Apply LocalityFeedForward
        x = x + self.drop_path(self.attn(self.norm1(x)))
        cls_token, x = torch.split(x, [1, N - 1], dim=1)
        x = x.transpose(1, 2).view(B, C, patch_size, patch_size)
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = torch.cat([cls_token, x], dim=1)
        return x

    def create_shifted_mask(self, patch_size, window_size, shift_size):
        mask = torch.zeros((1, patch_size, patch_size, 1))
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = mask.view(1, patch_size // window_size, window_size,
                                 patch_size // window_size, window_size, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, 1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

# New 2: Adjust LocalVisionTransformer for Dynamic Kernel Size
class LocalVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 act=3, reduction=4, wo_dp_conv=False, dp_first=False, window_size=7):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                         drop_path_rate, hybrid_backbone, norm_layer)

        if act == 1:
            act = 'relu6'
        elif act == 2:
            act = 'hs'
        elif act == 3:
            act = 'hs+se'
        elif act == 4:
            act = 'hs+eca'
        else:
            act = 'hs+ecah'

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act=act, reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first,
                window_size=window_size, kernel_size=7 if i < depth // 3 else 5 if i < 2 * depth // 3 else 3
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)


@register_model
def localvit_tiny_mlp6_act1(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=6, qkv_bias=True, act=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# reduction = 4
@register_model
def localvit_tiny_mlp4_act3_r4(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# reduction = 192
@register_model
def localvit_tiny_mlp4_act3_r192(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def localvit_small_mlp4_act3_r384(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, act=3, reduction=384,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
