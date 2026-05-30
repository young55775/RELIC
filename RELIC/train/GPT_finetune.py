import os
import argparse
import math
import shutil
import time
import pickle
from collections import OrderedDict, Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# 0. Configuration & Global Optimizations
# ==========================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ==========================================
# 1. Model Components
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return (self.weight.float() * x_norm).to(dtype)

class ProteinGPTConfig:
    def __init__(self, code_vocab_size=8192, num_classes=11,
                 embed_dim=768, vq_dim=384, layers=24, heads=12, dropout=0.1):
        self.code_vocab_size = code_vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.vq_dim = vq_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout

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
        cos = cos[position_ids].unsqueeze(2)
        sin = sin[position_ids].unsqueeze(2)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    cos_fp32 = cos_fp32.transpose(1, 2)
    sin_fp32 = sin_fp32.transpose(1, 2)
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q_fp32 * cos_fp32) + (rotate_half(q_fp32) * sin_fp32)
    k_embed = (k_fp32 * cos_fp32) + (rotate_half(k_fp32) * sin_fp32)
    return q_embed.type_as(q), k_embed.type_as(k)

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.heads, dropout=config.dropout, batch_first=True)
        self.q_norm = RMSNorm(config.embed_dim)
        self.k_norm = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        self.attn.out_proj.is_residual_projection = True
        self.mlp[2].is_residual_projection = True

    def forward(self, x, attn_mask=None, rope_cos=None, rope_sin=None):
        residual = x
        x_norm = self.ln1(x)
        q, k, v = F.linear(x_norm, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        B, L, _ = q.shape
        num_heads = self.attn.num_heads
        head_dim = q.shape[-1] // num_heads
        q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, num_heads, head_dim).transpose(1, 2)
        q_cond, q_seq = q[:, :, :1, :], q[:, :, 1:, :]
        k_cond, k_seq = k[:, :, :1, :], k[:, :, 1:, :]
        q_seq, k_seq = apply_rotary_pos_emb(q_seq, k_seq, rope_cos, rope_sin)
        q = torch.cat([q_cond, q_seq], dim=2)
        k = torch.cat([k_cond, k_seq], dim=2)
        if attn_mask is not None and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0, is_causal=False)
        x_attn = x_attn.transpose(1, 2).reshape(B, L, -1)
        x_attn = self.attn.out_proj(x_attn)
        x = residual + x_attn
        x = x + self.mlp(self.ln2(x))
        return x

class ProteinGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_idx = config.code_vocab_size
        self.seq_vocab_size = config.code_vocab_size + 5
        self.cond_embedding = nn.Embedding(config.num_classes + 1, config.embed_dim)
        self.seq_embedding = nn.Embedding(self.seq_vocab_size, config.embed_dim, padding_idx=self.pad_idx)
        self.rope = RotaryEmbedding(config.embed_dim // config.heads)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.layers)])
        self.ln_f = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, self.seq_vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
            if hasattr(module, "is_residual_projection") and module.is_residual_projection:
                scale = 1.0 / math.sqrt(2 * self.config.layers)
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attn_mask, pos_ids=None):
        cond_ids = input_ids[:, 0]
        seq_ids = input_ids[:, 1:]
        x_cond = self.cond_embedding(cond_ids).unsqueeze(1)
        x_seq = self.seq_embedding(seq_ids)
        x = torch.cat([x_cond, x_seq], dim=1)
        cos, sin = self.rope(x_seq, position_ids=pos_ids[:, 1:])
        for block in self.blocks:
            x = block(x, rope_cos=cos, rope_sin=sin, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, x

# ==========================================
# 2. Optimized Data Loading
# ==========================================
class LazyUnirefDataset(Dataset):
    def __init__(self, fpath):
        self.fpath = fpath
        self.data_infos = []
        self.file_handle = None
        self.labels = []
        index_file = fpath + ".labeled.idx"
        if os.path.exists(index_file):
            print(f"Loading cached labeled index from {index_file}...")
            with open(index_file, 'rb') as f:
                self.data_infos = pickle.load(f)
        else:
            print(f"Building labeled index for {fpath}...")
            self.data_infos = []
            with open(fpath, 'r') as f:
                offset = 0
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line: break
                    if line.startswith(">"):
                        try:
                            label_str = line.strip().replace(">", "")
                            label = int(label_str)
                            self.data_infos.append((offset, label))
                        except ValueError:
                            pass
                    _ = f.readline()
            print(f"Saving index to {index_file}...")
            with open(index_file, 'wb') as f:
                pickle.dump(self.data_infos, f)
        self.labels = [x[1] for x in self.data_infos]
        print(f"Loaded {len(self.data_infos)} sequences.")
        ctr = Counter(self.labels)
        print(f"Class Distribution: {dict(ctr)}")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.file_handle is None:
            self.file_handle = open(self.fpath, 'r')
        offset, label = self.data_infos[idx]
        self.file_handle.seek(offset)
        _header = self.file_handle.readline()
        seq_str = self.file_handle.readline().strip()
        try:
            if not seq_str: return [0], label
            codes = [int(x) for x in seq_str.split()]
            return codes, label
        except ValueError:
            return [0], label
    def __del__(self):
        if self.file_handle:
            self.file_handle.close()

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed package")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed package")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        labels = torch.tensor(dataset.labels, dtype=torch.long)
        class_counts = torch.bincount(labels)
        class_counts = class_counts.float() + 1e-6
        class_weights = 1.0 / class_counts
        self.weights = class_weights[labels]
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + 12345)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
    def __len__(self):
        return self.num_samples
    def set_epoch(self, epoch):
        self.epoch = epoch

class ConditionalCollate:
    def __init__(self, max_len, code_vocab_size):
        self.max_len = max_len
        self.pad_idx = code_vocab_size
        self.sos_idx = code_vocab_size + 1
        self.eos_idx = code_vocab_size + 2
        self.neg_inf = float('-inf')

    def __call__(self, batch):
        B = len(batch)
        in_ids_l, pos_l, tgt_l, labels_l = [], [], [], []
        for seq_codes, label in batch:
            seq_t = torch.tensor(seq_codes, dtype=torch.long)
            seq_part = torch.cat([torch.tensor([self.sos_idx]), seq_t, torch.tensor([self.eos_idx])])
            full_input = torch.cat([torch.tensor([label]), seq_part])
            pos_part = torch.arange(0, len(seq_part), dtype=torch.long)
            full_pos = torch.cat([torch.tensor([0]), pos_part])
            full_tgt = full_input.clone()
            full_tgt[0] = -100
            if len(full_input) > self.max_len:
                full_input = full_input[:self.max_len]
                full_pos = full_pos[:self.max_len]
                full_tgt = full_tgt[:self.max_len]
            in_ids_l.append(full_input)
            pos_l.append(full_pos)
            tgt_l.append(full_tgt)
            labels_l.append(label)
        b_ids = torch.full((B, self.max_len), self.pad_idx, dtype=torch.long)
        b_pos = torch.zeros((B, self.max_len), dtype=torch.long)
        b_tgt = torch.full((B, self.max_len), -100, dtype=torch.long)
        b_mask = torch.full((B, 1, self.max_len, self.max_len), self.neg_inf, dtype=torch.float32)
        b_labels = torch.tensor(labels_l, dtype=torch.long)
        for i in range(B):
            l = len(in_ids_l[i])
            b_ids[i, :l] = in_ids_l[i]
            b_pos[i, :l] = pos_l[i]
            b_tgt[i, :l] = tgt_l[i]
            causal_mask = torch.triu(torch.ones(l, l), diagonal=1).bool()
            valid_area = torch.zeros((l, l), dtype=torch.float32)
            valid_area.masked_fill_(causal_mask, self.neg_inf)
            b_mask[i, 0, :l, :l] = valid_area
        return b_ids, b_pos, b_mask, b_tgt, b_labels

# ==========================================
# 3. Scheduler Function
# ==========================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_ref = optimizer.param_groups[0]['initial_lr']
        min_ratio = min_lr / base_ref
        return min_ratio + (1.0 - min_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[name] = v
    return new_state_dict

# ==========================================
# 4. Fine-tuning Loop (Fixed Validation)
# ==========================================
def run_finetune(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # --- Dataset ---
    train_ds = LazyUnirefDataset(args.train_file)
    train_sampler = DistributedWeightedSampler(train_ds, rank=rank, num_replicas=world_size, replacement=True)
    collate_fn = ConditionalCollate(max_len=args.static_len, code_vocab_size=args.codebook_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)

    val_ds = LazyUnirefDataset(args.val_file) if args.val_file else None
    val_loader = None
    if val_ds:
        val_sampler = DistributedSampler(val_ds, rank=rank, shuffle=True) 
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                                collate_fn=collate_fn, num_workers=4, drop_last=False)

    # --- Model ---
    config = ProteinGPTConfig(
        code_vocab_size=args.codebook_size,
        num_classes=11,
        embed_dim=args.embed_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout
    )
    model = ProteinGPT(config).to(rank)

    if args.resume_from:
        if rank == 0: print(f"🔄 Loading pre-trained weights from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location='cpu')
        clean_ckpt = clean_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(clean_ckpt, strict=False)
        if rank == 0:
            print(f"⚠️ Missing (New Layers): {missing}")
            print(f"⚠️ Unexpected: {unexpected}")

    model = DDP(model, device_ids=[rank])

    # --- Optimizer ---
    cond_params_ids = list(map(id, model.module.cond_embedding.parameters()))
    base_params = filter(lambda p: id(p) not in cond_params_ids, model.module.parameters())
    optimizer_grouped_parameters = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.module.cond_embedding.parameters(), 'lr': args.lr * 10.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    min_lr = args.min_lr if args.min_lr is not None else 1e-6

    if rank == 0:
        print(f"\n🚀 Strategy: Warmup({args.warmup_steps}) -> Cosine Decay")
        print(f"   - Total Steps: {total_steps}")
        print(f"   - Base LR: {args.lr}")
        print(f"   - Save Checkpoint Every: {args.save_steps} steps")
        if val_loader:
            print(f"   - Validation: Random ~20% of data (Shuffle + Early Break)\n")

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps, min_lr)
    writer = SummaryWriter(log_dir=args.runs_dir) if rank == 0 else None
    global_step = 0
    max_val_steps = 0
    if val_loader:
        max_val_steps = int(len(val_loader) * 0.2)
        max_val_steps = max(10, max_val_steps) 

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader

        for ids, pos, mask, tgts, _ in pbar:
            ids, pos, mask, tgts = [x.to(rank, non_blocking=True) for x in [ids, pos, mask, tgts]]

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(ids, mask, pos_ids=pos)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tgts[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if rank == 0:
                base_lr_now = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{base_lr_now:.2e}")
                if global_step % 100 == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/LR_Base', base_lr_now, global_step)

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.save_dir, f"checkpoint_step_{global_step}.pth")
                    print(f"\n💾 Saving periodic checkpoint to {save_path} ...")
                    save_state = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(save_state, save_path)
                    weight_only_path = os.path.join(args.save_dir, f"weights_step_{global_step}.pth")
                    torch.save(model.module.state_dict(), weight_only_path)

            # --- Validation (Random 20%) ---
            if global_step % args.save_steps == 0 and val_loader:
                model.eval()
                val_sampler.set_epoch(global_step) 
                
                class_metrics = torch.zeros(11, 2).to(rank)
                val_steps_done = 0

                if rank == 0: print(f"🔍 Validating on ~20% of data ({max_val_steps} batches)...")

                with torch.no_grad():
                    for v_ids, v_pos, v_mask, v_tgts, v_labels in val_loader:
                        # 🔥 修改点 2: Early Break 实现 20% 采样
                        if val_steps_done >= max_val_steps:
                            break
                        val_steps_done += 1

                        v_ids, v_pos, v_mask, v_tgts, v_labels = [x.to(rank) for x in [v_ids, v_pos, v_mask, v_tgts, v_labels]]

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            v_logits, _ = model(v_ids, v_mask, pos_ids=v_pos)

                            shift_logits = v_logits[..., :-1, :].contiguous()
                            shift_labels = v_tgts[..., 1:].contiguous()

                            loss_tensor = F.cross_entropy(
                                shift_logits.transpose(1, 2),
                                shift_labels,
                                ignore_index=-100,
                                reduction='none'
                            )

                            seq_loss_sum = loss_tensor.sum(dim=1) 

                            valid_token_mask = (shift_labels != -100).float()
                            valid_token_count = valid_token_mask.sum(dim=1) + 1e-6 # 防止除0

                            loss_per_item = seq_loss_sum / valid_token_count

                            for i, lbl in enumerate(v_labels):
                                if lbl < 11:
                                    class_metrics[lbl, 0] += loss_per_item[i]
                                    class_metrics[lbl, 1] += 1.0

                dist.all_reduce(class_metrics, op=dist.ReduceOp.SUM)

                if rank == 0:
                    print(f"\n📊 [Step {global_step}] Validation Report (20% sampled):")
                    total_val_loss = 0.0
                    total_count = 0.0
                    for c in range(11):
                        count = class_metrics[c, 1].item()
                        if count > 0:
                            avg_loss = class_metrics[c, 0].item() / count
                            print(f"  Class {c}: Loss = {avg_loss:.4f} (Count: {int(count)})")
                            writer.add_scalar(f'Val/Class_{c}_Loss', avg_loss, global_step)
                            total_val_loss += class_metrics[c, 0].item()
                            total_count += count
                    
                    if total_count > 0:
                        overall_avg = total_val_loss / total_count
                        print(f"  👉 Overall Val Loss: {overall_avg:.4f}\n")
                        writer.add_scalar('Val/Overall_Loss', overall_avg, global_step)

                model.train()

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "finetune_final.pth"))
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--resume_from", type=str, required=True)
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--static_len", type=int, default=640)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="./finetune_checkpoints")
    parser.add_argument("--runs_dir", type=str, default="./finetune_runs")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    run_finetune(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), args)
