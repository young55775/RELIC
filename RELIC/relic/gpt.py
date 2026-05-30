import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.heads, dropout=config.dropout, batch_first=True)
        self.q_norm = RMSNorm(config.embed_dim)
        self.k_norm = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim), nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim), nn.Dropout(config.dropout)
        )
        self.attn.out_proj.is_residual_projection = True
        self.mlp[2].is_residual_projection = True

    def forward(self, x, attn_mask=None, rope_cos=None, rope_sin=None):
        residual = x
        x_norm = self.ln1(x)
        q, k, v = F.linear(x_norm, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        B, L, _ = q.shape
        num_heads = self.attn.num_heads
        head_dim = q.shape[-1] // num_heads
        q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, num_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        if attn_mask is not None and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x_attn = x_attn.transpose(1, 2).reshape(B, L, -1)
        x = residual + self.attn.out_proj(x_attn)
        x = x + self.mlp(self.ln2(x))
        return x

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x): return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ProteinGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_idx = config.code_vocab_size
        self.seq_vocab_size = config.code_vocab_size + 5
        self.seq_embedding = nn.Embedding(self.seq_vocab_size, config.embed_dim, padding_idx=self.pad_idx)
        self.rope = RotaryEmbedding(config.embed_dim // config.heads)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.layers)])
        self.ln_f = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, self.seq_vocab_size, bias=False)

    def forward(self, input_ids, pos_ids, attn_mask):
        x = self.seq_embedding(input_ids)
        cos, sin = self.rope(x, position_ids=pos_ids)

        for block in self.blocks:
            x = block(x, rope_cos=cos, rope_sin=sin, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, x
