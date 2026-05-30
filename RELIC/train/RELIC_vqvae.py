import os
import pickle as pkl
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import esm
import shutil
from collections import OrderedDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from esm.modules import TransformerLayer, ESM1bLayerNorm, RobertaLMHead
from tqdm import tqdm
from datetime import timedelta
import subprocess

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def copy_data_to_shm(original_dir, shm_dir, is_main_process):
    files_to_copy = [
        "train.fasta", "train.index.pkl",
        "val.fasta", "val.index.pkl",
        "test.fasta", "test.index.pkl"
    ]
    if is_main_process:
        os.makedirs(shm_dir, exist_ok=True)
        for file_name in files_to_copy:
            original_path = os.path.join(original_dir, file_name)
            shm_path = os.path.join(shm_dir, file_name)
            if not os.path.exists(shm_path):
                if os.path.exists(original_path):
                    shutil.copy2(original_path, shm_path)
    dist.barrier()

def create_index_if_needed(fasta_file, index_file, local_rank):
    if local_rank == 0:
        if not os.path.exists(index_file):
            try:
                cmd = ["grep", "--byte-offset", "^>", fasta_file]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                indices = []
                for line in process.stdout:
                    if not line: break
                    offset = int(line.split(':', 1)[0])
                    indices.append(offset)
                process.wait()
                if process.returncode != 0:
                    raise Exception("Grep failed")
            except Exception:
                indices = []
                with open(fasta_file, 'rb') as f:
                    if f.read(1) == b'>': indices.append(0)
                    f.seek(0)
                    chunk_size = 1024 * 1024 * 16
                    offset = 0
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk: break
                        pos = 0
                        while True:
                            try:
                                pos = chunk.index(b'\n>', pos)
                                indices.append(offset + pos + 1)
                                pos += 2
                            except ValueError:
                                break
                        offset += len(chunk)
                with open(index_file, 'wb') as out:
                    pkl.dump(indices, out)
    if dist.is_initialized():
        dist.barrier()

class IndexedUniRef(Dataset):
    def __init__(self, fasta_file, index_file, target_len):
        self.target_len = target_len
        self.fasta_file = fasta_file
        with open(index_file, 'rb') as f:
            self.index = pkl.load(f)
        self.f = None

    def _open_file_if_needed(self):
        if self.f is None:
            self.f = open(self.fasta_file, 'r')

    def _read_sequence_at_offset(self, offset):
        self._open_file_if_needed()
        self.f.seek(offset)
        header_line = self.f.readline().strip()
        if not header_line.startswith(">"):
            return "ERROR_ID", "X"
        seq_id = header_line[1:].split()[0].strip()
        seq_parts = []
        while True:
            line = self.f.readline()
            if not line or line.startswith('>'):
                break
            seq_parts.append(line.strip())
        full_seq = "".join(seq_parts)
        processed_seq = full_seq.replace('j', 'i').replace('J', 'I')
        return seq_id, processed_seq

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        offset = self.index[idx]
        try:
            seq_id, original_seq = self._read_sequence_at_offset(offset)
        except Exception:
            seq_id, original_seq = "ERROR_ID", "X"

        original_len = len(original_seq)
        is_complete_start = False
        is_complete_end = False
        processed_seq = ""
        if original_len > self.target_len:
            start_index = random.randint(0, original_len - self.target_len)
            processed_seq = original_seq[start_index : start_index + self.target_len]
            if start_index == 0: is_complete_start = True
            if (start_index + self.target_len) == original_len: is_complete_end = True
        else:
            max_start_percent = int(original_len * 0.1)
            max_start_abs = 30
            max_start_index = max(0, min(max_start_percent, max_start_abs))
            max_start_index = min(max_start_index, original_len - 1)
            max_start_index = max(0, max_start_index)
            if max_start_index > 0:
                start_index = random.randint(0, max_start_index)
            else:
                start_index = 0
            if start_index == 0: is_complete_start = True
            processed_seq = original_seq[start_index:]
            is_complete_end = True
        return seq_id, processed_seq, is_complete_start, is_complete_end

class SequenceCollateFunction:
    def __init__(self, alphabet, model_target_len):
        self.alphabet = alphabet
        self.model_target_len = model_target_len

    def __call__(self, batch):
        all_token_ids = []
        valid_batch = [item for item in batch if item[0] != "ERROR_ID"]

        for (seq_id, seq_str, is_start, is_end) in valid_batch:
            final_seq_parts = []
            if is_start: final_seq_parts.append('<cls>')
            final_seq_parts.append(seq_str)
            if is_end: final_seq_parts.append('<eos>')

            final_seq_str = "".join(final_seq_parts)
            token_ids = list(self.alphabet.encode(final_seq_str))

            if len(token_ids) < self.model_target_len:
                padding = [self.alphabet.padding_idx] * (self.model_target_len - len(token_ids))
                token_ids.extend(padding)
            elif len(token_ids) > self.model_target_len:
                token_ids = token_ids[:self.model_target_len]

            all_token_ids.append(torch.tensor(token_ids, dtype=torch.long))

        if not all_token_ids:
             dummy = torch.full((1, self.model_target_len), self.alphabet.padding_idx, dtype=torch.long)
             return dummy, dummy

        batch_tokens = torch.stack(all_token_ids, dim=0)
        batch_targets = batch_tokens.clone()
        batch_targets[batch_tokens == self.alphabet.padding_idx] = -100
        return batch_tokens, batch_targets

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class ConvUnetAttentionLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = ESM1bLayerNorm(embed_dim)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck_transformer = TransformerLayer(
            embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads,
            add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True,
        )
        self.unpool1 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.tconv1 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.unpool2 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _apply_norm(self, norm_layer, x_conv):
        x_permuted = x_conv.permute(0, 2, 1)
        x_normed = norm_layer(x_permuted)
        return x_normed.permute(0, 2, 1)

    def forward(self, x, self_attn_padding_mask=None, need_head_weights=False, return_bottleneck_only=False):
        x_residual = x
        x_conv = self.pre_norm(x).permute(1, 2, 0)

        h_conv1 = self.conv1(x_conv)
        skip1 = self.activation(self._apply_norm(self.norm1, h_conv1))
        h_pool1 = self.pool1(skip1)

        h_conv2 = self.conv2(h_pool1)
        skip2 = self.activation(self._apply_norm(self.norm2, h_conv2))
        h_pool2 = self.pool2(skip2)

        h_attn_in = h_pool2.permute(2, 0, 1)

        pooled_padding_mask = None
        if self_attn_padding_mask is not None:
            m = self_attn_padding_mask.float().unsqueeze(1)
            pooled_mask_1 = F.max_pool1d(m, kernel_size=2, stride=2)
            pooled_mask_2 = F.max_pool1d(pooled_mask_1, kernel_size=2, stride=2).bool()
            pooled_padding_mask = pooled_mask_2.squeeze(1)

        attn_out, attn_weights = self.bottleneck_transformer(
            h_attn_in, self_attn_padding_mask=pooled_padding_mask, need_head_weights=need_head_weights,
        )

        if return_bottleneck_only:
            return attn_out.permute(1, 2, 0)

        h_upsample_in = attn_out.permute(1, 2, 0)

        h_up1 = self.unpool1(h_upsample_in)
        h_up1_tconv = self.tconv1(h_up1 + skip2)
        h_up1 = self.activation(self._apply_norm(self.norm_t1, h_up1_tconv))

        h_up2 = self.unpool2(h_up1)
        h_up2_tconv = self.tconv2(h_up2 + skip1)
        h_up2 = self.activation(self._apply_norm(self.norm_t2, h_up2_tconv))

        x_out = h_up2.permute(2, 0, 1)

        if x_out.shape[0] != x_residual.shape[0]:
            diff = x_residual.shape[0] - x_out.shape[0]
            if diff > 0: x_out = F.pad(x_out, (0,0,0,0,0,diff))
            else: x_out = x_out[:x_residual.shape[0]]

        x = x_residual + self.dropout_layer(x_out)
        return x, attn_weights

class ESM2_ConvUnet_Encoder(nn.Module):
    def __init__(self, num_layers=12, embed_dim=560, attention_heads=20, alphabet="ESM-1b"):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        if not isinstance(alphabet, esm.data.Alphabet): alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.pad_idx = alphabet.padding_idx
        self.embed = nn.Embedding(len(alphabet), embed_dim, padding_idx=self.pad_idx)
        self.blocks = nn.ModuleList([ConvUnetAttentionLayer(embed_dim, 4*embed_dim, attention_heads) for _ in range(num_layers)])
        self.emb_layer_norm_after = ESM1bLayerNorm(embed_dim)

    def forward_phase2(self, tokens):
        mask = tokens.eq(self.pad_idx)
        x = self.embed(tokens)
        if mask is not None: x = x * (1 - mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        if not mask.any(): mask = None

        for i, layer in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                return layer(x, mask, return_bottleneck_only=True)
            else:
                output = layer(x, mask, return_bottleneck_only=False)
                if isinstance(output, tuple): x = output[0]
                else: x = output
        return x

class TransformerAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=12, num_layers=4):
        super().__init__()
        self.pre_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=output_dim,
                ffn_embed_dim=output_dim * 4,
                attention_heads=num_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True
            )
            for _ in range(num_layers)
        ])
        self.norm = ESM1bLayerNorm(output_dim)

    def forward(self, x, padding_mask=None):
        x = x.permute(2, 0, 1)
        x = self.pre_proj(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
        x = self.norm(x)
        x = x.permute(1, 2, 0)
        return x

class TransformerDecoderAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=20, num_layers=8):
        super().__init__()
        self.pre_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=output_dim,
                ffn_embed_dim=output_dim * 4,
                attention_heads=num_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True
            )
            for _ in range(num_layers)
        ])
        self.norm = ESM1bLayerNorm(output_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.pre_proj(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        x = x.permute(1, 2, 0)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs, mask_1d=None):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        inputs_norm = F.normalize(inputs, p=2, dim=2)
        embed_norm = F.normalize(self.embedding.weight, p=2, dim=1)

        if mask_1d is not None:
            valid_mask = ~mask_1d
            valid_count = torch.sum(valid_mask).float()
            flat_input = inputs_norm[valid_mask].contiguous()
        else:
            valid_count = float(inputs.size(0) * inputs.size(1))
            flat_input = inputs_norm.view(-1, self.embedding_dim)
            valid_mask = None

        distances = 2 - 2 * torch.matmul(flat_input, embed_norm.t())

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized_valid = torch.matmul(encodings, embed_norm)

        if quantized_valid.dtype != inputs.dtype:
             quantized_valid = quantized_valid.to(inputs.dtype)

        full_quantized = torch.zeros_like(inputs, device=inputs.device)
        if valid_mask is not None:
            mask_flat = valid_mask.view(-1)
            flat_output = torch.zeros(inputs.shape[0]*inputs.shape[1], self.embedding_dim, device=inputs.device, dtype=inputs.dtype)
            flat_output[mask_flat] = quantized_valid
            full_quantized = flat_output.view(input_shape)
        else:
            full_quantized = quantized_valid.view(input_shape)

        if self.training:
            with torch.no_grad():
                encodings_sum = torch.sum(encodings, dim=0)
                dw = torch.matmul(flat_input.t(), encodings).t().contiguous()

                if dist.is_initialized():
                    dist.all_reduce(encodings_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(dw, op=dist.ReduceOp.SUM)

                self._ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                self._ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = torch.sum(self._ema_cluster_size)
                cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

                raw_w = self._ema_w / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(F.normalize(raw_w, p=2, dim=1))

                dead_codes = self._ema_cluster_size < 1.0
                if dead_codes.any():
                    num_dead = dead_codes.sum().item()
                    rand_idx = torch.randperm(flat_input.shape[0], device=inputs.device)[:num_dead]
                    if len(rand_idx) < num_dead:
                         rand_idx = torch.randint(0, flat_input.shape[0], (num_dead,), device=inputs.device)
                    new_centers = flat_input[rand_idx].to(self.embedding.weight.dtype)
                    if dist.is_initialized():
                         dist.broadcast(new_centers, src=0)
                    self.embedding.weight.data[dead_codes] = new_centers
                    self._ema_w.data[dead_codes] = new_centers * self._ema_cluster_size[dead_codes].unsqueeze(1)

        if valid_mask is not None:
             e_loss = F.mse_loss(quantized_valid.detach(), flat_input, reduction='sum') / (valid_count + 1e-6)
        else:
             e_loss = F.mse_loss(full_quantized.detach(), inputs_norm, reduction='sum') / (valid_count + 1e-6)

        vq_commitment_loss = self.commitment_cost * e_loss
        latent_recon_loss = e_loss.detach()

        final_quantized_output = inputs_norm + (full_quantized - inputs_norm).detach()
        return vq_commitment_loss, final_quantized_output.permute(0, 2, 1).contiguous(), latent_recon_loss, encoding_indices

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_quantizers, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(num_quantizers)
        ])

    def forward(self, x, mask_1d=None):
        residual = x
        quantized_out = 0
        total_vq_loss = 0
        total_recon_loss = 0
        all_indices = []

        for layer in self.layers:
            vq_loss, quantized, recon_loss, indices = layer(residual, mask_1d)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_vq_loss += vq_loss
            total_recon_loss += recon_loss
            all_indices.append(indices.unsqueeze(-1))

        return total_vq_loss, quantized_out, total_recon_loss, torch.cat(all_indices, dim=-1)

class HybridDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads=20):
        super().__init__()
        self.global_attn_1_layers = nn.ModuleList([
            TransformerLayer(embed_dim, 4 * embed_dim, num_heads, add_bias_kv=False,
                            use_esm1b_layer_norm=True, use_rotary_embeddings=True)
            for _ in range(8)
        ])

        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            LayerNorm1d(embed_dim),
            nn.GELU()
        )
        self.global_attn_2 = nn.ModuleList([
            TransformerLayer(embed_dim, 4 * embed_dim, num_heads,
                                            add_bias_kv=False, use_esm1b_layer_norm=True,
                                            use_rotary_embeddings=True)
            for _ in range(8)
        ])
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            LayerNorm1d(embed_dim),
            nn.GELU()
        )
        self.final_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, super_tokens, mask_l4=None, mask_l2=None):
        x = super_tokens.permute(2, 0, 1)
        for layer in self.global_attn_1_layers:
            x, _ = layer(x, self_attn_padding_mask=mask_l4)
        x = x.permute(1, 2, 0)
        x = self.up1(x)
        x = x.permute(2, 0, 1)
        for layer in self.global_attn_2:
            x, _ = layer(x, self_attn_padding_mask=mask_l2)
        x = x.permute(1, 2, 0)
        x = self.up2(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits

class ProteinVQAutoencoder(nn.Module):
    def __init__(self, encoder, adapter, vq_layer, decoder_adapter, decoder):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.vq = vq_layer
        self.decoder_adapter = decoder_adapter
        self.decoder = decoder
        self.input_norm = nn.LayerNorm(560)
        self.pad_idx = encoder.pad_idx
        for param in self.encoder.parameters():
            param.requires_grad = False

    def downsample_mask(self, mask, scale):
        m = (~mask).float().unsqueeze(1)
        m = F.max_pool1d(m, kernel_size=scale, stride=scale, ceil_mode=False)
        return ~(m.bool().squeeze(1))

    def forward(self, tokens, vq_mix_ratio=1.0):
        B, L = tokens.shape
        remainder = L % 4
        if remainder != 0:
            pad_amt = 4 - remainder
            tokens = F.pad(tokens, (0, pad_amt), value=self.pad_idx)

        with torch.no_grad():
            encoded_output = self.encoder.forward_phase2(tokens)
            batch_size = tokens.shape[0]
            if encoded_output.shape[0] == batch_size and encoded_output.shape[1] == 560: super_tokens_frozen = encoded_output
            elif encoded_output.shape[1] == batch_size and encoded_output.shape[2] == 560: super_tokens_frozen = encoded_output.permute(1, 2, 0)
            elif encoded_output.shape[0] == batch_size and encoded_output.shape[2] == 560: super_tokens_frozen = encoded_output.permute(0, 2, 1)
            else: super_tokens_frozen = encoded_output.permute(1, 2, 0)

        normed_tokens = self.input_norm(super_tokens_frozen.permute(0, 2, 1)).permute(0, 2, 1)

        padding_mask = tokens.eq(self.pad_idx)
        L_feat = 1024
        mask_l2 = self.downsample_mask(padding_mask, 2)
        mask_l4 = self.downsample_mask(mask_l2, 2)

        super_tokens_trainable = self.adapter(normed_tokens, padding_mask=mask_l4)

        L_feat = super_tokens_trainable.shape[2]
        if mask_l4.shape[1] > L_feat: mask_l4 = mask_l4[:, :L_feat]
        elif mask_l4.shape[1] < L_feat: mask_l4 = F.pad(mask_l4, (0, L_feat - mask_l4.shape[1]), value=True)

        vq_commitment_loss, quantized_raw, _, encoding_indices = self.vq(super_tokens_trainable, mask_1d=mask_l4)

        if self.training:
            skip_prob = 1.0 - vq_mix_ratio
            batch_mask = torch.bernoulli(torch.full((B, 1, 1), skip_prob, device=tokens.device))
            decoder_input = batch_mask * super_tokens_trainable + (1 - batch_mask) * quantized_raw
        else:
            if vq_mix_ratio < 1.0:
                decoder_input = (1 - vq_mix_ratio) * super_tokens_trainable + vq_mix_ratio * quantized_raw
            else:
                decoder_input = quantized_raw

        restored_features = self.decoder_adapter(decoder_input)

        active_mask = (~mask_l4).unsqueeze(1)
        min_len = min(normed_tokens.shape[2], restored_features.shape[2])
        normed_slice = normed_tokens[:, :, :min_len]
        restored_slice = restored_features[:, :, :min_len]
        mask_slice = active_mask[:, :, :min_len]

        recon_sq_diff = (normed_slice - restored_slice) ** 2
        masked_sq_diff = recon_sq_diff * mask_slice.float()
        global_latent_recon_loss = masked_sq_diff.sum() / (mask_slice.float().sum() * 560.0 + 1e-6)

        L_mid = restored_features.shape[2] * 2
        if mask_l2.shape[1] > L_mid: mask_l2 = mask_l2[:, :L_mid]
        elif mask_l2.shape[1] < L_mid: mask_l2 = F.pad(mask_l2, (0, L_mid - mask_l2.shape[1]), value=True)

        logits = self.decoder(restored_features, mask_l4, mask_l2)
        return logits, vq_commitment_loss, global_latent_recon_loss, encoding_indices

class TrainingConfig:
    original_data_dir = "./clustered_split/"
    shm_data_dir = "/dev/shm/clustered_split_data"
    pretrained_ckpt = "/home/data/gzy/esm_unet_150M_generator/checkpoints_conv_unet_v3_fixed/esm2_conv_unet_v2_step_500000.pth"

    data_dir = shm_data_dir
    train_fasta_file = os.path.join(data_dir, "train.fasta")
    val_fasta_file = os.path.join(data_dir, "val.fasta")
    test_fasta_file = os.path.join(data_dir, "test.fasta")
    train_index_file = os.path.join(data_dir, "train.index.pkl")
    val_index_file = os.path.join(data_dir, "val.index.pkl")
    test_index_file = os.path.join(data_dir, "test.index.pkl")

    save_dir = './checkpoints_rq_vqvae_2layer_large'
    log_dir = './runs/rq_vqvae_2layer_large'

    num_layers = 12
    encoder_embed_dim = 560
    attention_heads = 20

    vq_embed_dim = 384
    codebook_size = 8192
    target_len = 1022
    learning_rate = 5e-5
    total_training_steps = 500000
    warmup_steps = 5000
    end_lr_factor = 0.1
    batch_size = 64
    grad_accum_steps = 1
    num_workers = 16
    log_every_n_steps = 100
    val_every_n_steps = 1000
    save_every_n_steps = 1000
    seed = 42

    vq_warmup_steps = 15000
    vq_warmup_start_value = 0.0

def count_parameters(model: nn.Module, is_main_process: bool):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print("=" * 50)
        print(f"Model Parameter Count: Total={total_params:,}, Trainable={trainable_params:,}")
        print("=" * 50)
    return total_params, trainable_params

@torch.no_grad()
def run_validation(model, val_loader, criterion, device, is_main_process, latent_weight=1.0, alphabet=None):
    model.eval()
    codebook_counts = torch.zeros(model.module.vq.layers[0].num_embeddings, device=device, dtype=torch.long)
    total_loss = 0
    total_recon_loss = 0
    total_vq_commitment_loss = 0
    total_latent_recon_loss = 0
    total_samples = 0

    val_iterator = val_loader
    if is_main_process: val_iterator = tqdm(val_loader, desc="Validating", ncols=120, leave=False)

    for batch in val_iterator:
        tokens, targets = batch
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_masked = targets.clone()
        targets_masked[targets == alphabet.padding_idx] = -100

        with autocast(dtype=torch.bfloat16):
            logits, vq_commitment_loss, latent_recon_loss, indices = model(tokens, vq_mix_ratio=1.0)

            min_len = min(logits.shape[1], targets.shape[1])
            logits_slice = logits[:, :min_len]
            targets_slice = targets_masked[:, :min_len]

            recon_loss = criterion(logits_slice.reshape(-1, len(alphabet)), targets_slice.reshape(-1))
            loss = recon_loss + vq_commitment_loss + latent_weight * latent_recon_loss

        indices_layer0 = indices[:, :, 0].flatten()
        batch_counts = torch.bincount(indices_layer0, minlength=model.module.vq.layers[0].num_embeddings)
        codebook_counts += batch_counts
        batch_size = tokens.size(0)
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_vq_commitment_loss += vq_commitment_loss.item() * batch_size
        total_latent_recon_loss += latent_recon_loss.item() * batch_size
        total_samples += batch_size

    dist.all_reduce(codebook_counts, op=dist.ReduceOp.SUM)
    num_codes_used = (codebook_counts > 0).sum().item()
    total_codes = model.module.vq.layers[0].num_embeddings
    utilization_rate = num_codes_used / total_codes

    if total_samples == 0:
        return {'total': 0.0, 'recon': 0.0, 'vq': 0.0, 'latent': 0.0, 'util': 0.0}

    avg_loss = total_loss / total_samples
    avg_recon = total_recon_loss / total_samples
    avg_vq = total_vq_commitment_loss / total_samples
    avg_latent = total_latent_recon_loss / total_samples

    if is_main_process:
        idx = 0
        pred_ids = torch.argmax(logits[idx], dim=-1)
        valid_len = min_len

        tgt_seq = ""
        for t in targets[idx, :valid_len]:
            val = t.item()
            if val == -100: continue
            if 0 <= val < len(alphabet):
                tok = alphabet.get_tok(val)
                if tok not in ['<pad>', '<mask>', '<cls>', '<eos>']: tgt_seq += tok

        pred_seq = ""
        valid_pred = pred_ids[1:-1] if valid_len > 2 else pred_ids
        for t in valid_pred:
            val = t.item()
            if 0 <= val < len(alphabet):
                tok = alphabet.get_tok(val)
                if tok not in ['<pad>', '<mask>', '<cls>', '<eos>']: pred_seq += tok

        print(f"\n[Val Sample] GT: {tgt_seq[:50]}...")
        print(f"[Val Sample] PR: {pred_seq[:50]}...")

    return {
        'total': avg_loss,
        'recon': avg_recon,
        'vq': avg_vq,
        'latent': avg_latent,
        'util': utilization_rate
    }

def get_vq_mix_ratio(step, total_steps, warmup_steps):
    if step >= warmup_steps: return 1.0

    progress = step / warmup_steps
    progress = max(0.0, min(1.0, progress))

    power = 0.2
    ratio = 1.0 - (1.0 - progress) ** (1.0 / power)
    return max(0.0, min(1.0, ratio))

def main():
    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    args = TrainingConfig()
    set_seed(args.seed)
    copy_data_to_shm(args.original_data_dir, args.shm_data_dir, is_main_process)
    create_index_if_needed(args.train_fasta_file, args.train_index_file, local_rank)
    create_index_if_needed(args.val_fasta_file, args.val_index_file, local_rank)
    create_index_if_needed(args.test_fasta_file, args.test_index_file, local_rank)

    writer = None
    if is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)

    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    train_set = IndexedUniRef(args.train_fasta_file, args.train_index_file, args.target_len)
    collate_fn = SequenceCollateFunction(alphabet, model_target_len=args.target_len + 2)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn, sampler=train_sampler, pin_memory=True)

    val_set = IndexedUniRef(args.val_fasta_file, args.val_index_file, args.target_len)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn, sampler=DistributedSampler(val_set, shuffle=False, seed=args.seed))

    encoder = ESM2_ConvUnet_Encoder(12, 560, 20, alphabet)
    if os.path.exists(args.pretrained_ckpt):
        sd = torch.load(args.pretrained_ckpt, map_location='cpu')
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k.replace("_orig_mod.", "").replace("module.", "").replace("base_model.", "")
            if "layers." in name: name = name.replace("layers.", "blocks.")
            if "embed_tokens" in name: name = name.replace("embed_tokens", "embed")
            new_sd[name] = v
        encoder.load_state_dict(new_sd, strict=False)
        if is_main_process: print("Encoder checkpoint loaded.")

    adapter = TransformerAdapter(input_dim=560, output_dim=384, num_heads=12, num_layers=4)

    vq_layer = ResidualVectorQuantizer(8192, 384, 1, 0.2, 0.99)
    decoder_adapter = TransformerDecoderAdapter(input_dim=384, output_dim=560, num_heads=20, num_layers=16)

    decoder = HybridDecoder(560, len(alphabet), num_heads=20)

    model = ProteinVQAutoencoder(encoder, adapter, vq_layer, decoder_adapter, decoder).to(device)

    count_parameters(model, is_main_process)

    model.decoder = torch.compile(model.decoder, mode="default")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,betas=(0.9,0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()

    LATENT_RECON_WEIGHT = 0
    global_step = 0
    epoch = 0

    while global_step < args.total_training_steps:
        epoch += 1
        train_sampler.set_epoch(epoch)
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120, leave=True) if is_main_process else train_loader

        for i, (tokens, targets) in enumerate(train_iterator):
            if tokens.shape[0] == 0: continue
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets_loss = targets.clone()
            targets_loss[targets == alphabet.padding_idx] = -100

            current_mix_ratio = get_vq_mix_ratio(global_step, args.total_training_steps, args.vq_warmup_steps)

            # Fixed weight 0.1 for latent loss
            current_latent_weight = LATENT_RECON_WEIGHT 

            with autocast(dtype=torch.bfloat16):
                logits, vq_loss_raw, latent_loss_raw, _ = model(tokens, vq_mix_ratio=current_mix_ratio)

                min_len = min(logits.shape[1], targets.shape[1])
                logits_slice = logits[:, :min_len]
                targets_slice = targets_loss[:, :min_len]

                recon_loss = criterion(logits_slice.reshape(-1, len(alphabet)), targets_slice.reshape(-1))

                loss = recon_loss + \
                       (vq_loss_raw * current_mix_ratio) + \
                       (latent_loss_raw * current_latent_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if is_main_process:
                train_iterator.set_postfix(
                    recon=f"{recon_loss.item():.4f}",
                    vq_c=f"{vq_loss_raw.item():.4f}",
                    lat_rec=f"{latent_loss_raw.item():.4f}",
                    mix=f"{current_mix_ratio:.2f}"
                )
                if writer and global_step % args.log_every_n_steps == 0:
                    writer.add_scalar('Loss/Train_Recon', recon_loss.item(), global_step)
                    writer.add_scalar('Loss/Train_VQ_Commit', vq_loss_raw.item(), global_step)
                    writer.add_scalar('Loss/Train_Latent_Recon_Raw', latent_loss_raw.item(), global_step)
                    writer.add_scalar('Training/VQ_Mix_Ratio', current_mix_ratio, global_step)

            if global_step % args.val_every_n_steps == 0:
                if is_main_process:
                    save_path = os.path.join(args.save_dir, f"vqvae_step_{global_step}.pth")
                    state = {'adapter': model.module.adapter.state_dict(), 'vq': model.module.vq.state_dict(), 'decoder_adapter': model.module.decoder_adapter.state_dict(), 'decoder': model.module.decoder.state_dict()}
                    torch.save(state, save_path)

                metrics = run_validation(model, val_loader, criterion, device, is_main_process, LATENT_RECON_WEIGHT, alphabet)
                if is_main_process:
                    print(f"\n[Step {global_step}] Recon: {metrics['recon']:.4f} Latent: {metrics['latent']:.4f} Codebook: {metrics['util']*100:.2f}%")
                    if writer:
                        writer.add_scalar('Loss/Val_Recon', metrics['recon'], global_step)
                        writer.add_scalar('Loss/Val_VQ', metrics['vq'], global_step)
                        writer.add_scalar('Loss/Val_Latent', metrics['latent'], global_step)
                        writer.add_scalar('Metrics/Codebook_Util', metrics['util'], global_step)
                model.train()

            if global_step >= args.total_training_steps: break
    cleanup_ddp()

if __name__ == "__main__":
    main()
