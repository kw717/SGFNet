import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np

from einops import rearrange
from timm.models.layers import to_2tuple


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace)

    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.act(out)
        return out


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class _NonLocalNd(nn.Module):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
            'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        # self.init_weights(**kwargs)

    # def init_weights(self, std=0.01, zeros_init=True):
    #     if self.mode != 'gaussian':
    #         for m in [self.g, self.theta, self.phi]:
    #             normal_init(m.conv, std=std)
    #     else:
    #         normal_init(self.g.conv, std=std)
    #     if zeros_init:
    #         if self.conv_out.norm_cfg is None:
    #             constant_init(self.conv_out.conv, 0)
    #         else:
    #             constant_init(self.conv_out.norm, 0)
    #     else:
    #         if self.conv_out.norm_cfg is None:
    #             normal_init(self.conv_out.conv, std=std)
    #         else:
    #             normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """

    def __init__(self, proj_type='pool', patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                                  groups=patch_size * patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
                                       nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.proj_type == 'conv':
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class LawinAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
                 patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size

        if self.head != 1:
            self.position_mixing = nn.ModuleList(
                [nn.Linear(patch_size * patch_size, patch_size * patch_size) for _ in range(self.head)])

    def forward(self, query, context):
        # x: [N, C, H, W]

        n = context.size(0)
        n, c, h, w = context.shape

        if self.head != 1:
            context = context.reshape(n, c, -1)
            context_mlp = []
            for hd in range(self.head):
                context_crt = context[:, (c // self.head) * (hd):(c // self.head) * (hd + 1), :]
                context_mlp.append(self.position_mixing[hd](context_crt))

            context_mlp = torch.cat(context_mlp, dim=1)
            context = context + context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output


class LawinHead(nn.Module):
    def __init__(self, embed_dim=768, use_scale=True, reduction=2, num_classes=9, **kwargs):
        super(LawinHead, self).__init__()
        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.in_channels = [64, 128, 320, 512]
        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        ConvModule(512, 512, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                                                   act_cfg=self.act_cfg))
        self.linear_c4 = MLP(input_dim=self.in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.short_path = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.cat = ConvModule(
            in_channels=512 * 5,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.low_level_fuse = ConvModule(
            in_channels=560,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.ds_8 = PatchEmbed(proj_type='pool', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='pool', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='pool', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)

        self.conv_seg = nn.Conv2d(512, num_classes, kernel_size=1)

    def get_context(self, x, patch_size):
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size * r,
                                 pw=patch_size * r, nh=h // patch_size, nw=w // patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context

    def cls_seg(self, feat):
        """Classify each pixel."""

        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):

        # inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # (n, c, 32, 32)
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))  # (n, c, 128, 128)
        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size,
                          nh=h // patch_size, nw=w // patch_size)

        output = []
        output.append(self.short_path(_c))
        output.append(F.interpolate(self.image_pool(_c),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False))

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size,
                                nh=h // patch_size, nw=w // patch_size)
            _output = F.interpolate(_output,
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False)
            output.append(_output)

        output = self.cat(torch.cat(output, dim=1))

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear', align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))

        output = self.cls_seg(output)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CaSa(nn.Module):
    def __init__(self, inchannels):
        super(CaSa, self).__init__()
        self.ca = ChannelAttention(in_planes=inchannels)
        self.sa = SpatialAttention()

    def forward(self, x):
        caout = self.ca(x)
        sa = self.sa(caout)
        out = torch.mul(caout, sa)
        return out


def dilationCBR(inchannls, d):
    return nn.Sequential(
        nn.Conv2d(in_channels=inchannls, out_channels=inchannls, kernel_size=3, dilation=d, padding=d),
        nn.BatchNorm2d(inchannls),
        nn.ReLU()
    )


def CBR3(inc):
    return nn.Sequential(
        nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1),
        nn.BatchNorm2d(inc),
        nn.ReLU()
    )


class CASPP(nn.Module):
    def __init__(self, inchannels):
        super(CASPP, self).__init__()
        self.conv2 = dilationCBR(inchannels, 2)
        self.conv4 = dilationCBR(inchannels, 4)
        self.conv6 = dilationCBR(inchannels, 6)
        self.outconv = nn.Sequential(
            nn.Conv2d(3 * inchannels, inchannels, kernel_size=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv2(x)
        x2 = self.conv4(x)
        x3 = self.conv6(x)
        o = torch.cat([x1, x2, x3], dim=1)
        o = self.outconv(o)
        return o


class SE_block(nn.Module):
    def __init__(self, inchannel, SEratio):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.liner_s = nn.Conv2d(inchannel * 4, inchannel * 4 // SEratio, kernel_size=1)
        self.liner_e = nn.Conv2d(inchannel * 4 // SEratio, inchannel * 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.hswish = nn.Hardswish()
        self.conv = nn.Conv2d(inchannel * 4, inchannel, kernel_size=1)
        self.fuseblock = fuseblock(inchannel)

    def forward(self, t, rgb, map):
        f = self.fuseblock(t, rgb)
        map = F.interpolate(map, t.size()[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([t, rgb, f], dim=1)
        # fuse = torch.mul(fuse,map)

        fusew = self.relu(self.liner_s(self.avg_pool(fuse)))
        fusew = self.hswish(self.liner_e(fusew))
        fuse = torch.mul(fusew, fuse)
        fuse = self.conv(fuse)
        fuse = fuse + rgb
        return fuse


class fuseblock(nn.Module):
    def __init__(self, inc):
        super(fuseblock, self).__init__()
        self.sa = SpatialAttention()
        self.sig = nn.Sigmoid()
        self.choose = base_choose_fuse(inc)

    def forward(self, rgb, t):
        att_r, att_t = self.choose(rgb, t)
        rgb = torch.mul(att_r, rgb)
        t = torch.mul(att_t, t)

        return torch.cat([rgb, t], dim=1)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=3, dilation=3, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class base_choose_fuse(nn.Module):
    def __init__(self, inc):
        super(base_choose_fuse, self).__init__()
        self.conv = nn.Conv2d(inc // 2, 2, 1)
        self.conv3 = depthwise_separable_conv(inc * 2, inc // 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, t):
        f = torch.cat([rgb, t], dim=1)

        f3 = self.conv3(f)

        f = self.conv(f3)
        f = self.softmax(f)

        att_r = f[:, 0, ...].unsqueeze(1)
        att_t = f[:, 1, ...].unsqueeze(1)

        return att_r, att_t


class SE_block2(nn.Module):
    def __init__(self, inchannel, SEratio):
        super(SE_block2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.liner_s = nn.Conv2d(inchannel * 4, inchannel * 4 // SEratio, kernel_size=1)
        self.liner_e = nn.Conv2d(inchannel * 4 // SEratio, inchannel * 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.hswish = nn.Hardswish()
        self.conv = nn.Conv2d(inchannel * 4, inchannel, kernel_size=1)
        self.fuseblock = fuseblock2()

    def forward(self, t, rgb, map):
        f = self.fuseblock(t, rgb)
        map = F.interpolate(map, t.size()[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([t, rgb, f], dim=1)
        fusew = self.relu(self.liner_s(self.avg_pool(fuse)))
        fusew = self.hswish(self.liner_e(fusew))
        fuse = torch.mul(fusew, fuse)
        fuse = self.conv(fuse)
        fuse = fuse + rgb
        return fuse


class fuseblock2(nn.Module):
    def __init__(self):
        super(fuseblock2, self).__init__()
        self.sa = SpatialAttention()
        self.sig = nn.Sigmoid()

    def forward(self, rgb, t):
        fuse = torch.mul(rgb, t)
        sa = self.sa(fuse)
        out = torch.mul(fuse, sa)
        out = self.sig(out)
        t = torch.mul(t, out)
        rgb = torch.mul(rgb, out)
        out = torch.cat([rgb, t], dim=1)
        return out


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        if d_hid > 50:
            cycle = 10
        elif d_hid > 5:
            cycle = 100
        else:
            cycle = 10000
        cycle = 10 if d_hid > 50 else 100
        return position / np.power(cycle, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class PosEncoding1D(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D, self).__init__()
        # print("use PosEncoding1D")
        self.sel_index = torch.tensor([0]).cuda()
        pos_enc = (get_sinusoid_encoding_table((128 // pos_rfactor) + 1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos  # B X H X W
        pos_h = pos_h // self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3)  # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()  # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda() // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x


class HANet_Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=4, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(HANet_Conv, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(mid_1_channel),
            nn.ReLU(inplace=True))

        if layer == 2:
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm1d(mid_2_channel),
                nn.ReLU(inplace=True))
            self.attention_third = nn.Sequential(
                nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))

        if self.pooling == 'mean':
            # print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((128 // pos_rfactor, 1))
        else:
            # print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((128 // pos_rfactor, 1))

        if pos_rfactor > 0:
            if is_encoding == 0:
                if self.pos_injection == 1:
                    pass
                elif self.pos_injection == 2:
                    pass
            elif is_encoding == 1:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            else:
                print("Not supported position encoding")
                exit()

    def forward(self, x, out, pos=None, return_attention=False, return_posmap=False, attention_loss=False):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        H = out.size(2)
        x1d = self.rowpool(x).squeeze(3)

        if pos is not None and self.pos_injection == 1:
            if return_posmap:
                x1d, pos_map1 = self.pos_emb1d_1st(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_1st(x1d, pos)

        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)
        x1d = self.attention_first(x1d)

        if pos is not None and self.pos_injection == 2:
            if return_posmap:
                x1d, pos_map2 = self.pos_emb1d_2nd(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_2nd(x1d, pos)

        x1d = self.attention_second(x1d)

        if self.layer == 3:
            x1d = self.attention_third(x1d)
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)
        else:
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)

        x1d = F.interpolate(x1d, size=H, mode='linear', align_corners=False)
        out = torch.mul(out, x1d.unsqueeze(3))

        if return_attention:
            if return_posmap:
                if self.pos_injection == 1:
                    pos_map = (pos_map1)
                elif self.pos_injection == 2:
                    pos_map = (pos_map2)
                return out, x1d, pos_map
            else:
                return out, x1d
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv2048 = ConvModule(2048, 1024, 3, 1, 1)
        self.conv1024 = ConvModule(1024, 512, 3, 1, 1)
        self.conv512 = ConvModule(512, 256, 3, 1, 1)
        self.conv256 = ConvModule(256, 64, 3, 1, 1)
        self.conv64 = ConvModule(64, 9, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, l5, l4, l3, l2, l1):
        l5 = self.up2(self.conv2048(l5))
        l4 = self.up2(self.conv1024(l5 + l4))
        l3 = self.up2(self.conv512(l4 + l3))
        l2 = self.up2(self.conv256(l3 + l2))
        l = l2 + l1
        out = self.up2(self.conv64(l2 + l1))
        return out, l


class BGALayer(nn.Module):

    def __init__(self, inc, guide_inc=64):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                inc, inc, kernel_size=3, stride=1,
                padding=1, groups=inc, bias=False),
            nn.BatchNorm2d(inc),
            nn.Conv2d(
                inc, guide_inc, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.right1 = nn.Sequential(
            nn.Conv2d(
                guide_inc, guide_inc, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(guide_inc),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                guide_inc, inc, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)

        right1 = self.right1(x_s)

        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = right1 * torch.sigmoid(left1)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out


class SGFNet(nn.Module):
    def __init__(self, n_classes=9):
        super(SGFNet, self).__init__()
        resnet_raw_model1 = models.resnet50(pretrained=True)
        resnet_raw_model2 = models.resnet50(pretrained=True)
        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        del resnet_raw_model1, resnet_raw_model2

        #######  CASA #######
        self.casa1 = CaSa(64)
        self.casa2 = CaSa(256)
        self.casa3 = CaSa(512)
        self.casa4 = CaSa(1024)
        self.casa5 = CaSa(2048)
        ####### MCDU ######
        self.SE_block1 = SE_block2(64, 16)
        self.SE_block2 = SE_block2(256, 16)
        self.SE_block3 = SE_block2(512, 16)
        self.SE_block4 = SE_block(1024, 16)
        self.SE_block5 = SE_block(2048, 16)
        ####### CSEU  #######
        self.haconv1 = BGALayer(64)
        self.haconv2 = BGALayer(256)
        self.haconv3 = BGALayer(512)
        self.haconv4 = BGALayer(1024)
        self.hanet12 = HANet_Conv(64, 256)
        self.hanet23 = HANet_Conv(256, 512)
        self.hanet34 = HANet_Conv(512, 1024)
        self.hanet45 = HANet_Conv(1024, 2048)
        ## Edge-aware Lawin ASPP ##
        self.lowconv = nn.Conv2d(256 + 64, 64, 1)
        self.compressconv1 = nn.Conv2d(512, 128, 1)
        self.compressconv2 = nn.Conv2d(1024, 320, 1)
        self.compressconv3 = nn.Conv2d(2048, 512, 1)
        self.lawin = LawinHead(num_classes=n_classes)
        self.edgeout = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.t_semout = FPN()
        self.binaryout = nn.Conv2d(9, 1, kernel_size=1)

    def forward(self, rgbinput, tin):
        rgb = rgbinput

        thermal = tin[:, :1, ...]

        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)
        t1 = thermal

        thermal = self.encoder_thermal_maxpool(thermal)
        thermal = self.encoder_thermal_layer1(thermal)
        t2 = thermal

        thermal = self.encoder_thermal_layer2(thermal)
        t3 = thermal

        thermal = self.encoder_thermal_layer3(thermal)
        t4 = thermal

        thermal = self.encoder_thermal_layer4(thermal)
        t5 = thermal
        t_sem, content_guide = self.t_semout(t5, t4, t3, t2, t1)
        binarymap = self.binaryout(t_sem)

        rgb = self.SE_block1(self.casa1(t1), rgb, binarymap)
        rgb1 = rgb
        rgb = self.encoder_rgb_maxpool(rgb)

        ######################################################################
        rgb = self.encoder_rgb_layer1(rgb)

        rgb = self.SE_block2(self.casa2(t2), rgb, binarymap)

        tempfuse = self.haconv1(rgb1, content_guide)

        rgb = self.hanet12(tempfuse, rgb)
        rgb2 = rgb

        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)

        rgb = self.SE_block3(self.casa3(t3), rgb, binarymap)

        tempfuse = self.haconv2(rgb2, content_guide)

        rgb = self.hanet23(tempfuse, rgb)
        rgb3 = rgb

        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)

        rgb = self.SE_block4(self.casa4(t4), rgb, binarymap)

        tempfuse = self.haconv3(rgb3, content_guide)

        rgb = self.hanet34(tempfuse, rgb)
        rgb4 = rgb
        ######################################################################
        rgb = self.encoder_rgb_layer4(rgb)

        fuse = self.SE_block5(self.casa5(t5), rgb, binarymap)

        tempfuse = self.haconv4(rgb4, content_guide)

        fuse = self.hanet45(tempfuse, fuse)
        ######################################################################
        rgb1 = F.interpolate(rgb1, rgb2.size()[2:], mode='bilinear', align_corners=False)
        low = torch.cat([rgb1, rgb2], dim=1)
        low = self.lowconv(low)
        edgeout = self.edgeout(low)
        rgb3 = self.compressconv1(rgb3)
        rgb4 = self.compressconv2(rgb4)
        fuse = self.compressconv3(fuse)
        input = [low, rgb3, rgb4, fuse]
        segout = self.lawin(input)
        segout = F.interpolate(segout, scale_factor=4, mode='bilinear', align_corners=False)
        edgeout = F.interpolate(edgeout, scale_factor=4, mode='bilinear', align_corners=False)

        t_sem = F.interpolate(t_sem, rgbinput.size()[2:], mode='bilinear', align_corners=False)

        return segout, edgeout, t_sem


if __name__ == '__main__':
    i = torch.randn(2, 3, 480, 640)
    i2 = torch.randn(2, 1, 480, 640)
    t = torch.randn(2, 64, 50, 50)
    rgb = torch.randn(2, 64, 50, 50)

    ca = SGFNet(9)
    out = ca(i, i2)
    for o in out:
        print(o.shape)
