import torch
import torch.nn as nn
from esm.modules import TransformerLayer, ESM1bLayerNorm
from collections import OrderedDict
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        self._update_cos_sin_cache(max_position_embeddings, device, dtype=torch.get_default_dtype())

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, position_ids):
        if position_ids.max() >= self.max_seq_len_cached:
            self._update_cos_sin_cache(position_ids.max() + 1024, x.device, x.dtype)
        cos = self.cos_cached.squeeze(0).squeeze(0)
        sin = self.sin_cached.squeeze(0).squeeze(0)
        return cos[position_ids].unsqueeze(2), sin[position_ids].unsqueeze(2)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_fp32, k_fp32 = q.float(), k.float()
    cos_fp32, sin_fp32 = cos.float(), sin.float()
    cos_fp32 = cos_fp32.transpose(1, 2)
    sin_fp32 = sin_fp32.transpose(1, 2)
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q_fp32 * cos_fp32) + (rotate_half(q_fp32) * sin_fp32)
    k_embed = (k_fp32 * cos_fp32) + (rotate_half(k_fp32) * sin_fp32)
    return q_embed.type_as(q), k_embed.type_as(k)

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x): return self.norm(x.transpose(1, 2)).transpose(1, 2)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '').replace('module.', '')
        if "cond_embedding" in name or "type_embedding" in name or "go_embedding" in name:
            continue

        new_state_dict[name] = v
    return new_state_dict


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def apply_repetition_penalty(logits, grouped_input_ids, penalty):
    score = torch.gather(logits, 1, grouped_input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, grouped_input_ids, score)
    return logits


def calc_banned_ngram_tokens(prev_input_ids, num_hyps, no_repeat_ngram_size, cur_len):
    if cur_len + 1 < no_repeat_ngram_size: return [[] for _ in range(num_hyps)]
    generated_ngrams = [{} for _ in range(num_hyps)]
    for idx in range(num_hyps):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hyp_idx):
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hyp_idx][start_idx:cur_len].tolist())
        return generated_ngrams[hyp_idx].get(ngram_idx, [])

    return [_get_generated_ngrams(h) for h in range(num_hyps)]
