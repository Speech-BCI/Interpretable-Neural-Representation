import torch
from torch import nn, einsum
import ast
from einops import rearrange, repeat, reduce
from modules.abstract_modules.spec_attn_modules import Patch_Attention, spec2tuple, Residual, create_2d_sin_embedding
from modules.abstract_modules.attn_utils import calculate_num_patches, adaptive_calculate_num_patches
from modules.abstract_modules.attn_utils import trunc_normal_
DEFAULT_DIM_HEAD = 64




class Spectrum_transformer(nn.Module):
    def __init__(self, spec_shape=(64, 128), patch_size=(8, 16), in_chans=32, embed_dim=256, overlap_ratio=0.5,class_token=True):
        super().__init__()
        spec_size = spec2tuple(spec_shape)
        patch_size = ast.literal_eval(patch_size)
        num_patches, stride_size, height_out, width_out = calculate_num_patches(spec_size, patch_size, overlap_ratio)

        self.spec_size = spec_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.class_token = class_token


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        self.proj_norm = nn.LayerNorm(embed_dim)
        if self.class_token:
            self.pos_embed = nn.Parameter(
                create_2d_sin_embedding(embed_dim, height_out, width_out).flatten(2).permute(0, 2, 1))
            # self.pos_embed = nn.Parameter(self.pos_embed * self.pos_scaling_factor)
            self.cls_token = nn.Parameter(torch.zeros(embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.pos_embed = nn.Parameter(
                create_2d_sin_embedding(embed_dim, height_out, width_out).flatten(2).permute(0, 2, 1))




    def forward(self, x_):
        hidden_states = self.proj(x_)
        b, d, f, t = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b d f t -> b (f t) d')

        hidden_states = hidden_states + self.pos_embed
        if self.class_token:
            cls_token = self.cls_token.repeat(b, 1, 1)
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        return hidden_states


class AttentionLayers(nn.Module):
    def __init__(self, attn_spec_shape, dim, depth, heads, dim_head, input_channel, causal=False, mask=None, talking_heads=True,
                 sparse_topk=None, use_entmax15=False, num_mem_kv=0, attn_dropout=0., ff_dropout=0., on_attn=False,
                 class_token = True, patch_size = (8, 8), overlap_ratio=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_norm = nn.LayerNorm(dim)
        self.enc_norm = nn.Sequential(nn.GELU(), nn.LayerNorm(dim))
        self.class_token = class_token
        self.patch_embedding = Spectrum_transformer(spec_shape = attn_spec_shape, embed_dim = dim, patch_size = patch_size, overlap_ratio=overlap_ratio,
                                                    class_token = class_token, in_chans = input_channel)
        self.residuals = Residual()

        for _ in range(depth):
            attention = Patch_Attention
            self.layers.append(nn.ModuleList([
                attention(dim, dim_head, heads, causal, mask, talking_heads, sparse_topk, use_entmax15, num_mem_kv, attn_dropout, on_attn),
            ]))

    def forward(self, x, context = None, mask = None, context_mask = None, mem = None, sinusoidal_emb = None, rel_pos = None, prev_attn = None):
        hidden_states = self.patch_embedding(x)
        # quantized_features, perplexity = self.quantizer(hidden_states[:, 1:,:], time_receptive_spec)
        enc_output = self.layer_norm(hidden_states)
        for attn_layer in self.layers:
            for attn in attn_layer:
                enc_output, intermediates = attn(enc_output)
        if self.class_token:
            pass
        else:
            # batch x seq_len x dim --> batch x dim x seq_len
            enc_output = enc_output.transpose(1, 2)

        return enc_output, intermediates


