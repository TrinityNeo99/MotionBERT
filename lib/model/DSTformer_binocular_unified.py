import copy

import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from collections import OrderedDict
from functools import partial
from itertools import repeat
from lib.model.drop import DropPath
from einops import rearrange, repeat
from lib.graph.pingpong_coco_bi import AdjMatrixGraph as Graph


def get_temporal_mask(max_len, window_size):
    assert max_len % window_size == 0, "the length of sequence can not divided by window size"

    # Initialize the matrix with zeros
    matrix = np.zeros((max_len, max_len))

    # Set the diagonal blocks to 1
    for i in range(0, max_len, window_size):
        matrix[i:i + window_size, i:i + window_size] = 1
    return matrix


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla',
                 isSpatialGraph=False, hop=1, isSpatialAttentionMoE=False, MoE_type="hop1234",
                 isTemporalAttentionMoE=True, temporal_MoE_type=[9, 27, 81, 243]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_count_s = None
        self.attn_count_t = None

        self.isSpatialGraph = isSpatialGraph
        if isSpatialGraph:
            graph = Graph()
            if hop == 1:
                graph_adjacency_matrix = torch.from_numpy(graph.A_binary)
            elif hop == 2:
                graph_adjacency_matrix = torch.from_numpy(graph.A_binary_with_I_2)
            elif hop == 3:
                graph_adjacency_matrix = torch.from_numpy(graph.A_binary_with_I_3)
            elif hop == 5:
                graph_adjacency_matrix = torch.from_numpy(graph.A_binary_with_I_5)
            elif hop == 6:
                graph_adjacency_matrix = torch.from_numpy(graph.A_binary_with_I_6)
            elif hop == "none":
                graph_adjacency_matrix = torch.ones((17, 17))
            else:
                raise Exception("Not Implementation")
            attention_mask = torch.where(graph_adjacency_matrix == 0, torch.tensor(-100000), torch.tensor(0.0))
            self.register_buffer('attention_mask', attention_mask)

        if isSpatialAttentionMoE:
            graph = Graph()
            if MoE_type == "hop123":
                self.num_experts = 3
                graph_adjacency_matrixs = []
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I))
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I_2))
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I_3))
                attention_masks = [torch.where(matrix == 0, torch.tensor(-100000), torch.tensor(0.0)) for matrix in
                                   graph_adjacency_matrixs]
                attention_masks = torch.stack(attention_masks, dim=0)
            elif MoE_type == "hop1234":
                self.num_experts = 4
                graph_adjacency_matrixs = []
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I))
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I_2))
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I_3))
                graph_adjacency_matrixs.append(torch.from_numpy(graph.A_binary_with_I_4))
                attention_masks = [torch.where(matrix == 0, torch.tensor(-100000), torch.tensor(0.0)) for matrix in
                                   graph_adjacency_matrixs]
                attention_masks = torch.stack(attention_masks, dim=0)
            else:
                raise Exception("Not Implementation")
            self.register_buffer('attention_masks', attention_masks)
            self.expert_linear = nn.Linear(dim, self.num_experts)
            self.expert_softmax = nn.Softmax(dim=-1)

        if isTemporalAttentionMoE:
            self.temporal_num_experts = len(temporal_MoE_type)
            temporal_matrixs = [torch.from_numpy(get_temporal_mask(243, window)) for window in temporal_MoE_type]
            temporal_attention_masks = [torch.where(matrix == 0, torch.tensor(-100000), torch.tensor(0.0)) for matrix in
                                        temporal_matrixs]
            temporal_attention_masks = torch.stack(temporal_attention_masks, dim=0)
            self.register_buffer('temporal_attention_masks', temporal_attention_masks)
            self.temporal_expert_linear = nn.Linear(dim, self.temporal_num_experts)
            self.temporal_expert_softmax = nn.Softmax(dim=-1)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        if self.mode == 'series':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'parallel':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x_t = self.forward_temporal(q, k, v, seqlen=seqlen)
            x_s = self.forward_spatial(q, k, v)

            alpha = torch.cat([x_s, x_t], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(B, 1, C, 2)
            alpha = alpha.softmax(dim=-1)
            x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
        elif self.mode == 'coupling':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_coupling(q, k, v, seqlen=seqlen)
        elif self.mode == 'vanilla':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'spatial_moe':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial_moe(x, q, k, v)
        elif self.mode == 'temporal_moe':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal_moe(x, q, k, v, seqlen=seqlen)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def reshape_T(self, x, seqlen=1, inverse=False):
        if not inverse:
            N, C = x.shape[-2:]
            x = x.reshape(-1, seqlen, self.num_heads, N, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, seqlen * N, C)  # (B, H, TN, c)
        else:
            TN, C = x.shape[-2:]
            x = x.reshape(-1, self.num_heads, seqlen, TN // seqlen, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, TN // seqlen, C)  # (BT, H, N, C)
        return x

    def forward_coupling(self, q, k, v, seqlen=8):
        BT, _, N, C = q.shape
        q = self.reshape_T(q, seqlen)
        k = self.reshape_T(k, seqlen)
        v = self.reshape_T(v, seqlen)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self.reshape_T(x, seqlen, inverse=True)
        x = x.transpose(1, 2).reshape(BT, N, C * self.num_heads)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.isSpatialGraph:
            attn = attn + self.attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def _forward_spatial_mask(self, q, k, v, attention_mask):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_spatial_moe(self, x, q, k, v):
        expert_outs = [self._forward_spatial_mask(q, k, v, self.attention_masks[i]) for i in range(self.num_experts)]
        expert_outs = torch.stack(expert_outs, dim=0)
        logit = self.expert_linear(x)
        expert_weights = self.expert_softmax(logit)
        expert_weights = expert_weights.permute(2, 0, 1).unsqueeze(-1)
        x = torch.sum(torch.mul(expert_outs, expert_weights), dim=0)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, N, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x

    def _forward_temporal_mask(self, q, k, v, attention_mask, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, N, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal_moe(self, x, q, k, v, seqlen=8):
        expert_outs = [self._forward_temporal_mask(q, k, v, self.temporal_attention_masks[i], seqlen) for i in
                       range(self.temporal_num_experts)]
        expert_outs = torch.stack(expert_outs, dim=0)
        logit = self.temporal_expert_linear(x)
        expert_weights = self.temporal_expert_softmax(logit)
        expert_weights = expert_weights.permute(2, 0, 1).unsqueeze(-1)
        x = torch.sum(torch.mul(expert_outs, expert_weights), dim=0)
        return x

    def count_attn(self, attn):
        attn = attn.detach().cpu().numpy()
        attn = attn.mean(axis=1)
        attn_t = attn[:, :, 1].mean(axis=1)
        attn_s = attn[:, :, 0].mean(axis=1)
        if self.attn_count_s is None:
            self.attn_count_s = attn_s
            self.attn_count_t = attn_t
        else:
            self.attn_count_s = np.concatenate([self.attn_count_s, attn_s], axis=0)
            self.attn_count_t = np.concatenate([self.attn_count_t, attn_t], axis=0)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()
        # assert 'stage' in st_mode
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="spatial")
        self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="temporal_moe")

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, seqlen=1):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == 'stage_para':
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                #             x_s, x_t: [BF, J, dim]
                alpha = torch.cat([x_s, x_t], dim=-1)
                BF, J = alpha.shape[:2]
                # alpha = alpha.mean(dim=1, keepdim=True)
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_st")
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        # self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, dim_out, global_pool=''):
        self.dim_out = dim_out
        self.head = nn.Linear(self.dim_feat, dim_out) if dim_out > 0 else nn.Identity()

    def forward(self, x, return_rep=False):
        B, F, J, C = x.shape
        alphas = []
        x = x.reshape(-1, J, C)
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        if return_rep:
            return x
        # x = self.head(x)
        x = x.reshape(B, F, J, C)
        return x

    def get_representation(self, x):
        return self.forward(x, return_rep=True)


class Unified_Binocular(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True, use_decoder=False, decoder_dim_feat=256,
                 encoder_left_right_fuse=False, multi_task_head=False, shared_interpreter=False):
        super().__init__()
        self.num_keypoints = num_joints

        self.joints_embed = nn.Linear(dim_in, dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.pos_embed_left = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.pos_embed_right = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.monocular_embed = nn.Parameter(torch.zeros(1, 1, dim_feat))
        self.binocular_embed_left = nn.Parameter(torch.zeros(1, 1, dim_feat))
        self.binocular_embed_right = nn.Parameter(torch.zeros(1, 1, dim_feat))
        self.binocular_embed = nn.Parameter(torch.zeros(1, 1, dim_feat))
        self.self2self_embed = nn.Parameter(torch.zeros(1, 1, dim_feat))

        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_left, std=.02)
        trunc_normal_(self.pos_embed_right, std=.02)
        trunc_normal_(self.monocular_embed, std=.02)
        trunc_normal_(self.binocular_embed_left, std=.02)
        trunc_normal_(self.binocular_embed_right, std=.02)
        trunc_normal_(self.binocular_embed, std=.02)
        trunc_normal_(self.self2self_embed, std=.02)

        self.encoder = DSTformer(dim_in=dim_in, dim_out=3, dim_feat=dim_feat, dim_rep=dim_rep,
                                 depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                 num_joints=num_joints, maxlen=maxlen,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                 attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, att_fuse=att_fuse)

        self.use_decoder = use_decoder
        if use_decoder:
            self.pool = nn.MaxPool2d
            self.decoder = DSTformer(dim_in=dim_in, dim_out=3, dim_feat=decoder_dim_feat, dim_rep=dim_rep,
                                     depth=2, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     num_joints=num_joints, maxlen=maxlen,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                     attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                     norm_layer=norm_layer, att_fuse=att_fuse)

            self.keypoint_monocular_3d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            self.keypoint_binocular_left_2d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            self.keypoint_binocular_right_2d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            self.keypoint_binocular_3d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            self.keypoint_binocular_spatial_3d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            self.keypoint_self2self_2d = nn.Parameter(torch.zeros(num_joints, dim_feat))
            if dim_feat != decoder_dim_feat:
                self.downsample = nn.Linear(dim_feat, decoder_dim_feat)
                dim_feat = decoder_dim_feat
            else:
                self.downsample = nn.Identity()

        self.multi_task_head = multi_task_head
        if multi_task_head:
            self.left2right_head = nn.Linear(dim_feat, dim_out)
            self.right2left_head = nn.Linear(dim_feat, dim_out)
            self.monocular_head = nn.Linear(dim_feat, dim_out)
            self.binocular_head = nn.Linear(dim_feat, dim_out)
            self.self2self_head = nn.Linear(dim_feat, dim_out)

        self.shared_interpreter = shared_interpreter
        if self.shared_interpreter:
            self.interpreter2d = nn.Linear(dim_rep, dim_out)  # the 2d confidence is the last channel
            self.interpreter3d = nn.Linear(dim_rep, dim_out)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.encoder_left_right_fuse = encoder_left_right_fuse
        if encoder_left_right_fuse:
            self.left_right_fusion_map = nn.Linear(num_joints * dim_feat, 2)

        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def forward(self, x, type="monocular"):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        x = self.joints_embed(x)
        if type == "binocular_spatial":
            x[:, :J // 2, :] += self.pos_embed_left
            x[:, J // 2:, :] += self.pos_embed_right
        else:
            x = x + self.pos_embed

        # task embedding
        if type == "self2self":
            x += self.self2self_embed
        elif type == "left2right":
            x += self.binocular_embed_left
        elif type == "right2left":
            x += self.binocular_embed_right
        elif type == "monocular":
            x += self.monocular_embed
        elif type == "binocular":
            x += self.binocular_embed
        elif type == "binocular_spatial":
            x += self.binocular_embed
        else:
            raise Exception("Undefined task type")
        _, J, C = x.shape

        # temporal embedding
        if type == "binocular":
            x = x.reshape(-1, F, J, C)
            f = F // 2
            x[:, :f, :, :] += self.temp_embed[:, :f, :, :]
            x[:, f:, :, :] += self.temp_embed[:, :f, :, :]
        else:
            x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :]
        # x = x.reshape(-1, J, C)
        x = self.pos_drop(x)
        x = self.encoder(x)

        if self.use_decoder:
            if type == "self2self":
                keypoint_tokens = repeat(self.keypoint_self2self_2d, 'j c -> b f j c', b=B, f=F)
            elif type == "left2right":
                keypoint_tokens = repeat(self.keypoint_binocular_right_2d, 'j c -> b f j c', b=B, f=F)
            elif type == "right2left":
                keypoint_tokens = repeat(self.keypoint_binocular_left_2d, 'j c -> b f j c', b=B, f=F)
            elif type == "monocular":
                keypoint_tokens = repeat(self.keypoint_monocular_3d, 'j c -> b f j c', b=B, f=F)
            elif type == "binocular":
                keypoint_tokens = repeat(self.keypoint_binocular_3d, 'j c -> b f j c', b=B, f=F // 2)
            elif type == "binocular_spatial":
                keypoint_tokens = repeat(self.keypoint_binocular_spatial_3d, 'j c -> b f j c', b=B, f=F)
            else:
                raise Exception("Undefined task type")
            x = torch.cat((keypoint_tokens, x), dim=2)  # along the spatial dimention
            x = self.downsample(x)
            x = self.decoder(x)
            if type == "binocular":
                x = x[:, :F // 2, :, :]
            else:
                x = x[:, :, :self.num_keypoints, :]
        else:
            if type == "binocular":
                x_left = x[:, :F // 2, :, :]
                x_right = x[:, F // 2:, :, :]
                x = (x_left + x_right) * 0.5
            elif type == "binocular_spatial":
                x_left = x[:, :, :J // 2, :]
                x_right = x[:, :, J // 2:, :]
                if self.encoder_left_right_fuse:
                    _x_left = rearrange(x_left, 'b t v c -> b t (v c)')
                    fusion_weights = self.left_right_fusion_map(_x_left)
                    fusion_weights = torch.softmax(fusion_weights, dim=-1)
                    x = x_left * fusion_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1) + x_right * fusion_weights[:, :,
                                                                                                 1].unsqueeze(
                        -1).unsqueeze(-1)
                else:
                    x = (x_left + x_right) * 0.5

        x = self.pre_logits(x)  # [B, F, J, dim_feat]
        logits = x

        if self.multi_task_head:
            if type == "left2right":
                x = self.left2right_head(x)
            elif type == "right2left":
                x = self.right2left_head(x)
            elif type == "binocular" or type == "binocular_spatial":
                x = self.binocular_head(x)
            elif type == "monocular":
                x = self.monocular_head(x)
            elif type == "self2self":
                x = self.self2self_head(x)
            else:
                raise Exception("Undefined task type")
        elif self.shared_interpreter:
            # xxxx
            if type in ["self2self", "left2right", "right2left"]:
                x = self.interpreter2d(x)
            elif type in ["monocular", "binocular", "binocular_spatial"]:
                x = self.interpreter3d(x)
            else:
                raise Exception("Undefined task type")
        else:
            raise Exception("Undefined head settings")
        return x, logits
