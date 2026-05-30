import os
import argparse
import math
import shutil
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import GradScaler 

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
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class ProteinGPTConfig:
    def __init__(self, code_vocab_size=8192, go_vocab_size=1000,
                 embed_dim=768, vq_dim=384, layers=24, heads=12, dropout=0.1,
                 use_conditional=False, use_vq_loss=False):
        self.code_vocab_size = code_vocab_size
        self.go_vocab_size = go_vocab_size
        self.embed_dim = embed_dim
        self.vq_dim = vq_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.use_conditional = use_conditional
        self.use_vq_loss = use_vq_loss

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
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        if attn_mask is not None and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        x_attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0, is_causal=False
        )
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
        self.seq_embedding = nn.Embedding(self.seq_vocab_size, config.embed_dim, padding_idx=self.pad_idx)

        if config.use_conditional:
            self.go_embedding = nn.Embedding(config.go_vocab_size, config.embed_dim, padding_idx=0)
            self.type_embedding = nn.Embedding(2, config.embed_dim)

        if config.use_vq_loss:
            self.register_buffer("frozen_vq_codebook", torch.zeros(config.code_vocab_size, config.vq_dim))
            self.feature_projection = nn.Linear(config.embed_dim, config.vq_dim)

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

    def forward(self, seq_ids, go_ids, type_ids, pos_ids, attn_mask):
        x = self.seq_embedding(seq_ids)

        if self.config.use_conditional:
            is_seq = type_ids.bool().unsqueeze(-1)
            x_go = self.go_embedding(go_ids)
            x = torch.where(is_seq, x, x_go)
            x = x + self.type_embedding(type_ids)

        cos, sin = self.rope(x, position_ids=pos_ids)
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
        self.offsets = []
        self.file_handle = None
        index_file = fpath + ".idx"
        if os.path.exists(index_file):
            print(f"Loading cached index from {index_file}...")
            self.offsets = torch.load(index_file)
        else:
            print(f"Building index for {fpath}...")
            offsets = []
            with open(fpath, 'r') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line: break
                    if not line.startswith(">"):
                        offsets.append(offset)
            self.offsets = offsets
            print(f"Saving index to {index_file}...")
            torch.save(self.offsets, index_file)
        print(f"Loaded {len(self.offsets)} sequences (Lazy Mode).")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self.file_handle is None:
            self.file_handle = open(self.fpath, 'r')
        offset = self.offsets[idx]
        self.file_handle.seek(offset)
        line = self.file_handle.readline()
        seq_str = line.strip()
        try:
            if not seq_str: return [0]
            codes = [int(x) for x in seq_str.split()]
            return codes
        except ValueError:
            return [0]
    def __del__(self):
        if self.file_handle:
            self.file_handle.close()

class UnconditionalCollate:
    def __init__(self, max_len, code_vocab_size):
        self.max_len = max_len
        self.pad_idx = code_vocab_size
        self.sos_idx = code_vocab_size + 1
        self.eos_idx = code_vocab_size + 2
        self.neg_inf = float('-inf')

    def __call__(self, batch):
        B = len(batch)
        in_ids_l, type_l, pos_l, tgt_l = [], [], [], []
        for seq_codes in batch:
            seq_t = torch.tensor(seq_codes, dtype=torch.long)
            full_ids = torch.cat([
                torch.tensor([self.sos_idx]),
                seq_t,
                torch.tensor([self.eos_idx])
            ])
            L = len(full_ids)
            t_type = torch.ones(L, dtype=torch.long)
            t_pos = torch.arange(1, L + 1, dtype=torch.long)
            t_target = full_ids.clone()
            t_target[0] = -100 # Mask SOS target

            if len(full_ids) > self.max_len:
                full_ids = full_ids[:self.max_len]
                t_type = t_type[:self.max_len]
                t_pos = t_pos[:self.max_len]
                t_target = t_target[:self.max_len]

            in_ids_l.append(full_ids)
            type_l.append(t_type)
            pos_l.append(t_pos)
            tgt_l.append(t_target)

        b_ids = torch.full((B, self.max_len), self.pad_idx, dtype=torch.long)
        b_type= torch.zeros((B, self.max_len), dtype=torch.long)
        b_pos = torch.zeros((B, self.max_len), dtype=torch.long)
        b_tgt = torch.full((B, self.max_len), -100, dtype=torch.long)
        b_mask = torch.full((B, 1, self.max_len, self.max_len), self.neg_inf, dtype=torch.float32)

        for i in range(B):
            l = len(in_ids_l[i])
            b_ids[i, :l] = in_ids_l[i]
            b_type[i, :l] = type_l[i]
            b_pos[i, :l] = pos_l[i]
            b_tgt[i, :l] = tgt_l[i]
            causal_mask = torch.triu(torch.ones(l, l), diagonal=1).bool()
            valid_area = torch.zeros((l, l), dtype=torch.float32)
            valid_area.masked_fill_(causal_mask, self.neg_inf)
            b_mask[i, 0, :l, :l] = valid_area

        return b_ids, torch.zeros_like(b_ids), b_type, b_pos, b_mask, b_tgt

# ==========================================
# 3. Helpers
# ==========================================
def compute_hybrid_loss(logits, hidden_states, target_ids, model_module, args):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

    if args.lambda_feat == 0.0 or not model_module.config.use_vq_loss:
        return loss_ce, loss_ce.item(), 0.0

    shift_hidden = hidden_states[..., :-1, :].contiguous()
    valid_mask = (shift_labels >= 0) & (shift_labels < args.codebook_size)
    if valid_mask.sum() == 0: return loss_ce, loss_ce.item(), 0.0

    pred_features = model_module.feature_projection(shift_hidden)
    safe_targets = shift_labels.clone()
    safe_targets[~valid_mask] = 0
    gt_features = F.embedding(safe_targets, model_module.frozen_vq_codebook)

    loss_feat = F.mse_loss(pred_features[valid_mask], gt_features[valid_mask])
    return loss_ce + args.lambda_feat * loss_feat, loss_ce.item(), loss_feat.item()

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================================
# 4. Main Training Loop
# ==========================================
def run_pretrain(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 1. Dataset
    train_ds = LazyUnirefDataset(args.train_file)
    val_ds = LazyUnirefDataset(args.val_file) if (args.val_file and not args.overfit) else None

    train_sampler = DistributedSampler(train_ds, rank=rank, shuffle=True)
    collate_fn = UnconditionalCollate(max_len=args.static_len, code_vocab_size=args.codebook_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=0 if args.overfit else 64,
                              pin_memory=True, drop_last=True)

    val_loader = None
    if val_ds:
        val_sampler = DistributedSampler(val_ds, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                                collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=False)

    current_dropout = 0.0 if args.overfit else args.dropout
    current_weight_decay = 0.0 if args.overfit else 0.01
    current_lr = 1e-3 if args.overfit else args.lr

    config = ProteinGPTConfig(
        code_vocab_size=args.codebook_size,
        go_vocab_size=args.dummy_go_vocab_size,
        embed_dim=args.embed_dim,
        vq_dim=args.vq_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=current_dropout,
        use_conditional=False,
        use_vq_loss=(args.lambda_feat > 0.0)
    )
    model = ProteinGPT(config).to(rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n" + "="*40)
        print("🤖 Model Statistics (Fixed Stability Mode)")
        print("="*40)
        print(f"Total Parameters     : {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"Layers               : {config.layers} (Scaled Init Enabled)")
        print(f"Embed Dim            : {config.embed_dim}")
        print(f"Scaler               : Removed (BF16 Native)")
        print("="*40 + "\n")

    print(f"Rank {rank}: Compiling model (Max Performance)...")
    model = torch.compile(
        model,
        mode="max-autotune"
    )

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay, betas=(0.9, 0.95), eps=1e-5)

    if args.overfit:
        if rank == 0:
            print("\n" + "="*40 + "\n🚨 ENABLED ONE-CLICK OVERFIT MODE 🚨\n" + "="*40)

        iterator = iter(train_loader)
        batch_data = next(iterator)
        seqs, gos, types, pos, mask, tgts = [x.to(rank, non_blocking=True) for x in batch_data]

        model.train()
        for step in range(1000):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, hidden_states = model(seqs, gos, types, pos, mask)
                loss, ce_v, _ = compute_hybrid_loss(logits, hidden_states, tgts, model.module, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scaler.update()

            if rank == 0: print(f"[Overfit Step {step:04d}] Loss: {loss.item():.6f}")
        dist.destroy_process_group()
        return

    # === Normal Training Loop ===
    total_steps = len(train_loader) * args.epochs
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    writer = SummaryWriter(log_dir=args.runs_dir) if rank == 0 else None
    global_step = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader

        for seqs, gos, types, pos, mask, tgts in pbar:
            seqs, gos, types, pos, mask, tgts = [x.to(rank, non_blocking=True) for x in [seqs, gos, types, pos, mask, tgts]]

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, hidden_states = model(seqs, gos, types, pos, mask)
                loss, ce_v, feat_v = compute_hybrid_loss(logits, hidden_states, tgts, model.module, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            # scaler.update()

            scheduler.step()
            global_step += 1

            if rank == 0 and global_step % args.log_freq == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_v:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if rank == 0 and global_step % args.save_freq == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, f"step_{global_step}.pth"))

            if val_loader and global_step % args.val_freq == 0:
                model.eval()
                val_loss, count = torch.tensor(0.0).to(rank), torch.tensor(0.0).to(rank)
                with torch.no_grad():
                    for i, (v_seqs, v_gos, v_types, v_pos, v_mask, v_tgts) in enumerate(val_loader):
                        if i >= 50: break
                        v_seqs, v_gos, v_types, v_pos, v_mask, v_tgts = [x.to(rank, non_blocking=True) for x in [v_seqs, v_gos, v_types, v_pos, v_mask, v_tgts]]
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            v_logits, v_hidden = model(v_seqs, v_gos, v_types, v_pos, v_mask)
                            _, v_ce, _ = compute_hybrid_loss(v_logits, v_hidden, v_tgts, model.module, args)
                        val_loss += v_ce
                        count += 1
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                avg_val = val_loss / count
                if rank == 0:
                    pbar.write(f"📊 [Step {global_step}] Val CE: {avg_val.item():.4f}")
                    writer.add_scalar('Val/CE', avg_val.item(), global_step)
                model.train()

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "pretrain_final.pth"))
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--dummy_go_vocab_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default="./pretrain_checkpoints")
    parser.add_argument("--runs_dir", type=str, default="./pretrain_runs")
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--freeze_steps", type=int, default=5000)
    parser.add_argument("--val_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--lambda_feat", type=float, default=0.0)
    parser.add_argument("--static_len", type=int, default=640)

    # Model Architecture
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--vq_dim", type=int, default=384)

    parser.add_argument("--overfit", action="store_true")

    args = parser.parse_args()
    run_pretrain(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), args)
