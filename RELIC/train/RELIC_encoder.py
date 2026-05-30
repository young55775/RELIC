import os
import argparse
import pickle as pkl
import random
import math
from typing import List, Tuple, Optional, Union
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import esm
import shutil

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from esm.modules import TransformerLayer, ESM1bLayerNorm, RobertaLMHead
from esm.multihead_attention import MultiheadAttention
from tqdm import tqdm
import datetime

def setup_ddp():
    dist.init_process_group(backend="nccl",timeout=datetime.timedelta(days=1))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def cleanup_ddp():
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

def mask_tokens(tokens: torch.Tensor, alphabet: esm.Alphabet, mlm_probability=0.15):
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=tokens.device)

    special_token_ids = [
        alphabet.get_idx(tok) if hasattr(alphabet, 'get_idx') else alphabet.tok_to_idx[tok]
        for tok in alphabet.all_special_tokens
    ]
    special_tokens_tensor = torch.tensor(special_token_ids, dtype=torch.long, device=tokens.device)

    special_tokens_mask = torch.isin(labels, special_tokens_tensor)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=tokens.device)).bool() & masked_indices
    tokens[indices_replaced] = alphabet.mask_idx

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=tokens.device)).bool() & masked_indices & ~indices_replaced

    random_words = torch.randint(alphabet.padding_idx + 1, len(alphabet), labels.shape, dtype=torch.long, device=tokens.device)
    tokens[indices_random] = random_words[indices_random]

    return tokens, labels

def get_gpu_memory_usage(device: torch.device) -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        return f"GPU {device}: {mem_alloc_mb:.0f} MB"
    except Exception as e:
        return f"Failed to get memory: {e}"


class IndexedUniRef(Dataset):
    def __init__(self, fasta_file: str, index_file: str, target_len: int):
        self.target_len = target_len
        self.fasta_file = fasta_file

        if not os.path.exists(fasta_file) or not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FASTA file or index file not found. "
                f"Checked: {fasta_file} and {index_file}. "
            )

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
            raise IOError(f"Offset {offset} does not point to a FASTA header.")

        seq_id = header_line[1:].split()[0].strip()

        seq_parts = []
        while True:
            current_pos = self.f.tell()
            line = self.f.readline()
            if not line or line.startswith('>'):
                break
            seq_parts.append(line.strip())

        full_seq = "".join(seq_parts)
        processed_seq = full_seq.replace('j', 'i').replace('J', 'I')
        return seq_id, processed_seq

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Tuple[str, str, bool, bool]:
        offset = self.index[idx]

        try:
            seq_id, original_seq = self._read_sequence_at_offset(offset)
        except Exception as e:
            print(f"Warning: Failed to read sequence at index {idx} (offset {offset}). Error: {e}")
            seq_id, original_seq = "ERROR_ID", "X"
        original_len = len(original_seq)
        is_complete_start = False
        is_complete_end = False
        processed_seq = ""
        if original_len > self.target_len:
            start_index = random.randint(0, original_len - self.target_len)
            processed_seq = original_seq[start_index : start_index + self.target_len]
            if start_index == 0:
                is_complete_start = True
            if (start_index + self.target_len) == original_len:
                is_complete_end = True
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
            if start_index == 0:
                is_complete_start = True
            processed_seq = original_seq[start_index:]
            is_complete_end = True
        return seq_id, processed_seq, is_complete_start, is_complete_end

class MLMCollateFunction:
    def __init__(self, alphabet, model_target_len: int):
        self.alphabet = alphabet
        self.model_target_len = model_target_len

    def __call__(self, batch: List[Tuple[str, str, bool, bool]]):
        all_token_ids = []
        valid_batch = [item for item in batch if item[0] != "ERROR_ID"]

        for (seq_id, seq_str, is_start, is_end) in valid_batch:
            token_ids_unpadded = self.alphabet.encode(seq_str)
            token_ids = list(token_ids_unpadded)

            if is_start:
                token_ids.insert(0, self.alphabet.cls_idx)
            if is_end:
                token_ids.append(self.alphabet.eos_idx)

            current_len = len(token_ids)

            if current_len < self.model_target_len:
                padding = [self.alphabet.padding_idx] * (self.model_target_len - current_len)
                token_ids.extend(padding)
            elif current_len > self.model_target_len:
                token_ids = token_ids[:self.model_target_len]

            all_token_ids.append(torch.tensor(token_ids, dtype=torch.long))

        if not all_token_ids:
             dummy_tok = torch.full((1, self.model_target_len), self.alphabet.padding_idx, dtype=torch.long)
             return dummy_tok

        batch_tokens = torch.stack(all_token_ids, dim=0)
        return batch_tokens


class ConvUnetAttentionLayer(Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        self.pre_norm = ESM1bLayerNorm(embed_dim)
        self.activation = nn.GELU()

        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck_transformer = TransformerLayer(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            attention_heads=attention_heads,
            add_bias_kv=False,
            use_esm1b_layer_norm=True,
            use_rotary_embeddings=True,
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
        x_restored = x_normed.permute(0, 2, 1)
        return x_restored

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x_residual = x
        x_normed = self.pre_norm(x)
        x_conv = x_normed.permute(1, 2, 0)

        h_conv1 = self.conv1(x_conv)
        skip1 = self.activation(self._apply_norm(self.norm1, h_conv1))
        h_pool1 = self.pool1(skip1)

        h_conv2 = self.conv2(h_pool1)
        skip2 = self.activation(self._apply_norm(self.norm2, h_conv2))
        h_pool2 = self.pool2(skip2)

        h_attn_in = h_pool2.permute(2, 0, 1)

        pooled_padding_mask = None
        if self_attn_padding_mask is not None:
            mask_for_pooling = self_attn_padding_mask.float().unsqueeze(1)
            pooled_mask_1 = F.max_pool1d(mask_for_pooling, kernel_size=2, stride=2)
            pooled_mask_2 = F.max_pool1d(pooled_mask_1, kernel_size=2, stride=2).bool()
            pooled_padding_mask = pooled_mask_2.squeeze(1)

        attn_out, attn_weights = self.bottleneck_transformer(
            h_attn_in,
            self_attn_padding_mask=pooled_padding_mask,
            need_head_weights=need_head_weights,
        )

        h_upsample_in = attn_out.permute(1, 2, 0)

        h_up1 = self.unpool1(h_upsample_in)
        h_up1_tconv = self.tconv1(h_up1 + skip2)
        h_up1 = self.activation(self._apply_norm(self.norm_t1, h_up1_tconv))

        h_up2 = self.unpool2(h_up1)
        h_up2_tconv = self.tconv2(h_up2 + skip1)
        h_up2 = self.activation(self._apply_norm(self.norm_t2, h_up2_tconv))

        x_out = h_up2.permute(2, 0, 1)

        x = x_residual + self.dropout_layer(x_out)

        return x, attn_weights

class ESM2_ConvUnet(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b"
        # 移除了 token_dropout 参数
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                ConvUnetAttentionLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    dropout=0.1,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False):
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(tokens)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        layer_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)

        if self.num_layers in repr_layers:
            hidden_representations[self.num_layers] = x

        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            result["attentions"] = attentions
        return result


class TrainingConfig:
    original_data_dir = "/home/guozy/data/clustered_split/"
    shm_data_dir = "/dev/shm/clustered_split_data"

    data_dir = shm_data_dir

    train_fasta_file = os.path.join(data_dir, "train.fasta")
    val_fasta_file = os.path.join(data_dir, "val.fasta")
    test_fasta_file = os.path.join(data_dir, "test.fasta")

    train_index_file = os.path.join(data_dir, "train.index.pkl")
    val_index_file = os.path.join(data_dir, "val.index.pkl")
    test_index_file = os.path.join(data_dir, "test.index.pkl")

    log_dir = './runs/esm2_conv_unet_v4_pure'
    save_dir = './checkpoints_conv_unet_v4_pure'

    num_layers = 12
    embed_dim = 560
    attention_heads = 20

    target_len = 1022

    learning_rate = 4e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_epsilon = 1e-8
    weight_decay = 0.01

    total_training_steps = 500000
    warmup_steps = 5000
    end_lr_factor = 0.1

    batch_size = 240
    grad_accum_steps = 1
    num_workers = 32

    log_every_n_steps = 100
    val_every_n_steps = 10000

    seed = 42


def count_parameters(model: nn.Module, is_main_process: bool):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process:
        print("=" * 50)
        print(f"Model Parameter Count:")
        print(f"  Total Params: {total_params:,}")
        print(f"  Trainable Params: {trainable_params:,}")
        print(f"  (Using {total_params / 1_000_000:.2f}M parameters)")
        print("=" * 50)

    return total_params, trainable_params


def run_validation(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    alphabet_size: int,
    is_main_process: bool,
    alphabet: esm.Alphabet  
) -> float:

    model.eval()
    total_loss = 0

    eval_progress_bar = None
    if is_main_process:
        eval_progress_bar = tqdm(
            data_loader,
            desc=f"  Evaluating... ",
            leave=True,
            ncols=100
        )
    else:
        eval_progress_bar = data_loader

    with torch.no_grad():
        for batch_tokens in eval_progress_bar:
            if batch_tokens.shape[0] == 0:
                continue

            batch_tokens = batch_tokens.to(device, non_blocking=True)
            masked_tokens, mlm_labels = mask_tokens(batch_tokens, alphabet)

            with autocast(dtype=torch.bfloat16):
                outputs = model(masked_tokens)
                mlm_logits = outputs["logits"]

                loss_mlm = loss_fn(
                    mlm_logits.view(-1, alphabet_size),
                    mlm_labels.view(-1)
                )

            total_loss += loss_mlm.item()

    avg_loss = total_loss / len(data_loader)

    model.train()
    return avg_loss


def copy_data_to_shm(original_dir, shm_dir, is_main_process):
    files_to_copy = [
        "train.fasta", "train.index.pkl",
        "val.fasta", "val.index.pkl",
        "test.fasta", "test.index.pkl"
    ]

    if is_main_process:
        print(f"Target shared memory directory: {shm_dir}")
        os.makedirs(shm_dir, exist_ok=True)

        copy_progress = tqdm(files_to_copy, desc="Copying to /dev/shm", ncols=100, leave=False)
        for file_name in copy_progress:
            original_path = os.path.join(original_dir, file_name)
            shm_path = os.path.join(shm_dir, file_name)

            copy_progress.set_postfix_str(f"{file_name}")

            if not os.path.exists(shm_path):
                if os.path.exists(original_path):
                    print(f"Starting copy: {original_path} -> {shm_path}")
                    shutil.copy2(original_path, shm_path)
                    print(f"Finished copy: {file_name}")
                else:
                    print(f"Warning: Source file not found, skipping: {original_path}")
            else:
                print(f"File already exists in shm, skipping copy: {file_name}")
        print("Data preparation in shared memory complete.")

    dist.barrier()


def main():

    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    args = TrainingConfig()
    set_seed(args.seed)

    copy_data_to_shm(
        original_dir=args.original_data_dir,
        shm_dir=args.shm_data_dir,
        is_main_process=is_main_process
    )

    writer = None
    if is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"--- DDP Initialized: {world_size} processes. ---")
        print(f"--- Logs will be saved to: {args.log_dir} ---")
        print(f"--- Checkpoints will be saved to: {args.save_dir} ---")

    if not torch.cuda.is_available():
        if is_main_process:
            print("!!! WARNING: CUDA not available, training on CPU !!!")

    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    alphabet_size = len(alphabet)

    try:
        if not os.path.exists(args.train_index_file):
            if is_main_process:
                print(f"Error: Training index file not found: {args.train_index_file}")
            raise FileNotFoundError

        if is_main_process:
            if not os.path.exists(args.val_index_file):
                print(f"Warning: Validation index file not found: {args.val_index_file}")
            if not os.path.exists(args.test_index_file):
                print(f"Warning: Test index file not found: {args.test_index_file}")

        dist.barrier()

    except FileNotFoundError:
        cleanup_ddp()
        return

    if is_main_process:
        print("Index files found. Proceeding with lazy loading from shared memory.")

    model_target_len = args.target_len + 2

    if is_main_process:
        print(f"--- Key Check ---")
        print(f"Model: ESM2_ConvUnet (v4 - PURE)")
        print(f"Config target_len (L_raw): {args.target_len} -> (Model): {model_target_len}")
        print(f"Per-GPU Batch: {args.batch_size}, Accum Steps: {args.grad_accum_steps}, World: {world_size}")
        print(f"Global Batch Size per Step: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"--- Check Complete ---")

    train_set = IndexedUniRef(
        fasta_file=args.train_fasta_file,
        index_file=args.train_index_file,
        target_len=args.target_len
    )

    collate_fn = MLMCollateFunction(alphabet, model_target_len=model_target_len)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True, seed=args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True
    )

    val_loader = None
    test_loader = None
    if is_main_process:
        val_set = IndexedUniRef(
            fasta_file=args.val_fasta_file,
            index_file=args.val_index_file,
            target_len=args.target_len
        )
        test_set = IndexedUniRef(
            fasta_file=args.test_fasta_file,
            index_file=args.test_index_file,
            target_len=args.target_len
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True
        )

    if is_main_process:
        print(f"Loading custom model ESM2_ConvUnet (v4 - PURE)...")

    model = ESM2_ConvUnet(
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        attention_heads=args.attention_heads,
        alphabet=alphabet
    ).to(device)

    total_params, _ = count_parameters(model, is_main_process)

    if is_main_process:
        print("Compiling model with torch.compile()... (this may take a moment)")

    model = torch.compile(model,options={"max_autotune":True})

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process:
        print("Configuring Optimizer and LR Scheduler...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        decay_duration = int(args.total_training_steps * 0.9)
        decay_end_step = args.warmup_steps + decay_duration
        if current_step > decay_end_step:
            return args.end_lr_factor

        progress = float(current_step - args.warmup_steps) / float(max(1, decay_duration))
        return 1.0 - (1.0 - args.end_lr_factor) * progress

    scheduler = LambdaLR(optimizer, lr_lambda)
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()

    if is_main_process:
        print("--- Starting Training ---")

    global_step = 0
    epoch = 0

    while global_step < args.total_training_steps:

        epoch += 1
        train_sampler.set_epoch(epoch)

        epoch_logs = {}
        model.train()
        total_mlm_loss_epoch = 0
        num_update_steps_epoch = 0
        val_losses_this_epoch = []

        train_iterator = train_loader
        if is_main_process:
            train_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                ncols=120,
                leave=True
            )

        for i, batch_tokens in enumerate(train_iterator):

            if batch_tokens.shape[0] == 0:
                if is_main_process:
                    print(f"Warning: Skipping empty batch (batch index {i}).")
                continue

            batch_tokens = batch_tokens.to(device, non_blocking=True)
            masked_tokens, mlm_labels = mask_tokens(batch_tokens, alphabet)

            with autocast(dtype=torch.bfloat16):
                outputs = model(masked_tokens)
                mlm_logits = outputs["logits"]

                loss_mlm = mlm_loss_fn(
                    mlm_logits.view(-1, alphabet_size),
                    mlm_labels.view(-1)
                )

                loss_to_backward = loss_mlm / args.grad_accum_steps

            scaler.scale(loss_to_backward).backward()

            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):

                dist.all_reduce(loss_mlm, op=dist.ReduceOp.AVG)
                current_loss = loss_mlm.item()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process:
                    total_mlm_loss_epoch += current_loss
                    num_update_steps_epoch += 1

                    current_lr = optimizer.param_groups[0]['lr']
                    postfix_data = {
                        'loss': f"{current_loss:.4f}",
                        'lr': f"{current_lr:.2e}",
                        'step': f"{global_step}/{args.total_training_steps}"
                    }
                    train_iterator.set_postfix(postfix_data)
                    train_iterator.set_description(f"Epoch {epoch}")

                    if writer:
                        writer.add_scalar('Loss/Step_Train_MLM', current_loss, global_step)
                        writer.add_scalar('Learning_Rate/Step', current_lr, global_step)


                if global_step % args.log_every_n_steps == 0 and is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    train_iterator.write("-" * 80)
                    train_iterator.write(
                        f"[Step {global_step:6d}] (Epoch {epoch}) - Step Train MLM Loss (Avg): {current_loss:.4f} - LR: {current_lr:.2e}"
                    )
                    if device.type == 'cuda':
                        mem_report = get_gpu_memory_usage(device)
                        train_iterator.write(f"[Memory Usage] {mem_report}")
                    train_iterator.write("-" * 80)

                if global_step % args.val_every_n_steps == 0 and is_main_process:
                    train_iterator.write(f"\n[Step {global_step:6d}] --- Starting Validation ---")

                    avg_val_loss = run_validation(
                        model, val_loader, mlm_loss_fn, device, alphabet_size, is_main_process, alphabet
                    )
                    val_losses_this_epoch.append(avg_val_loss)

                    train_iterator.write(f"[Step {global_step:6d}] --- Validation Finished --- Avg Val MLM Loss: {avg_val_loss:.4f}")

                    if writer:
                        writer.add_scalar('Loss/Step_Validation', avg_val_loss, global_step)

                    train_iterator.write(f"[Step {global_step:6d}] --- Saving Checkpoint ---")
                    save_path = os.path.join(args.save_dir, f"esm2_conv_unet_v4_pure_step_{global_step}.pth")
                    torch.save(model.module.state_dict(), save_path)
                    train_iterator.write(f"[Step {global_step:6d}] --- Checkpoint Saved to: {save_path}\n")

                if global_step >= args.total_training_steps:
                    if is_main_process:
                        train_iterator.write(f"--- Reached max training steps ({args.total_training_steps}). Stopping training. ---")
                    break

        if is_main_process:
            avg_epoch_loss = total_mlm_loss_epoch / num_update_steps_epoch if num_update_steps_epoch > 0 else 0
            epoch_logs['avg_train_mlm_loss'] = avg_epoch_loss
            if val_losses_this_epoch:
                avg_val_loss_for_epoch = np.mean(val_losses_this_epoch)
                epoch_logs['avg_val_mlm_loss'] = avg_val_loss_for_epoch

            print(f"Epoch {epoch} complete: Avg Train Loss: {avg_epoch_loss:.4f}")

        if global_step >= args.total_training_steps:
            break

    if is_main_process:
        print("--- Training Finished ---")
        print("\n" + "="*50)
        print(f"--- Training complete ({global_step} steps), running final test set evaluation ---")
        print("="*50)

        avg_test_loss = run_validation(
            model, test_loader, mlm_loss_fn, device, alphabet_size, is_main_process, alphabet
        )

        print("\n" + "="*50)
        print(f"--- Final Test Set MLM Loss: {avg_test_loss:.4f} ---")
        print("="*50)

        if writer:
            print("Logging hyperparameters and final metrics to TensorBoard...")
            hparam_dict = {k: v for k, v in vars(TrainingConfig).items() if not k.startswith('__') and isinstance(v, (str, int, float, bool))}
            hparam_dict["total_params"] = total_params
            metric_dict = {"final_test_loss": avg_test_loss}
            writer.add_hparams(hparam_dict, metric_dict)
            writer.close()

    cleanup_ddp()

if __name__ == "__main__":
    main()
