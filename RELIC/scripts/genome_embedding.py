import argparse
import math
import os
from collections import OrderedDict

import esm
import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from esm.modules import TransformerLayer, ESM1bLayerNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm


# ================= Model =================
class ConvUnetAttentionLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout=0.1):
        super().__init__()

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
        return norm_layer(x_conv.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x, self_attn_padding_mask=None, return_bottleneck=False):
        x_conv = self.pre_norm(x).permute(1, 2, 0)
        h_conv1 = self.conv1(x_conv)
        skip1 = self.activation(self._apply_norm(self.norm1, h_conv1))
        h_pool1 = self.pool1(skip1)
        h_conv2 = self.conv2(h_pool1)
        skip2 = self.activation(self._apply_norm(self.norm2, h_conv2))
        h_pool2 = self.pool2(skip2)
        attn_in = h_pool2.permute(2, 0, 1)
        pooled_mask = None
        if self_attn_padding_mask is not None:
            m = self_attn_padding_mask.float().unsqueeze(1)
            pooled_mask = F.max_pool1d(
                F.max_pool1d(m, 2, 2), 2, 2
            ).bool().squeeze(1)

        attn_out, _ = self.bottleneck_transformer(attn_in, self_attn_padding_mask=pooled_mask)

        if return_bottleneck:
            return attn_out.transpose(0, 1)

        h_upsample_in = attn_out.permute(1, 2, 0)
        h_up1 = self.activation(self._apply_norm(self.norm_t1, self.tconv1(self.unpool1(h_upsample_in) + skip2)))
        h_up2 = self.activation(self._apply_norm(self.norm_t2, self.tconv2(self.unpool2(h_up1) + skip1)))
        output = x + self.dropout_layer(h_up2.permute(2, 0, 1))
        return output


class ConvUnetExtractor(nn.Module):
    def __init__(self, num_layers=12, embed_dim=560, attention_heads=20, alphabet="ESM-1b"):
        super().__init__()
        self.num_layers = num_layers

        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)

        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx

        self.embed_tokens = nn.Embedding(len(alphabet), embed_dim, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList([
            ConvUnetAttentionLayer(embed_dim, 4 * embed_dim, attention_heads)
            for _ in range(num_layers)
        ])

    def forward(self, tokens):
        padding_mask = tokens.eq(self.padding_idx)
        x = self.embed_tokens(tokens)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                return layer(x, self_attn_padding_mask=padding_mask, return_bottleneck=True)
            else:
                x = layer(x, self_attn_padding_mask=padding_mask, return_bottleneck=False)
        return x


# ================= FASTA =================
def gen_pid_uniprot(desc):
    fields = desc.split("|")
    upid = fields[1]
    ff = fields[2].split(" ")
    gn = list(filter(lambda f: f.startswith("GN="), ff))
    if len(gn) > 0:
        return f"{gn[0][3:]}_{upid}"
    else:
        return f"{ff[0]}_{upid}"


def split_seq(seq, max_len=1022, overlap=256):
    chunks = []

    start = 0
    segid = 0
    step = max_len - overlap

    while start < len(seq):
        end = min(start + max_len, len(seq))
        chunks.append((segid, seq[start:end]))
        segid = segid + 1
        if end >= len(seq):
            break
        start += step
    return chunks


def read_fasta(path):
    records = []

    with open(path) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_id = gen_pid_uniprot(record.description)
            seq = str(record.seq)
            if seq[-1] == '*':
                seq = seq[:-1]

            chunked = split_seq(seq)
            for seg_id, sub_seq in chunked:
                records.append((f'{seq_id}_{seg_id}', sub_seq))

    return records


# ================= Dataset =================
class FastaDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch, alphabet):
    batch_converter = alphabet.get_batch_converter()
    labels = [x[0] for x in batch]
    seqs = [x[1] for x in batch]
    data = list(zip(labels, seqs))
    _, _, batch_tokens = batch_converter(data)

    original_token_length = batch_tokens.shape[1]
    target_token_length = math.ceil(original_token_length / 4) * 4
    target_seq_length = [math.ceil((len(seq) + 2) / 4) for seq in seqs]

    if target_token_length > original_token_length:
        batch_tokens = F.pad(
            batch_tokens,
            (0, target_token_length - original_token_length),
            value=alphabet.padding_idx
        )

    return labels, seqs, batch_tokens, target_seq_length


# ================= DDP =================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_fasta", required=True, help="Genome fasta file")
    parser.add_argument("-p", "--param_path", required=True, help="Parameter file")
    parser.add_argument("-o", "--output_h5", default='./output.h5', help="Output h5 file (default: output.h5)")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size (default: 16)")

    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    print(f"Loading model on cuda:{local_rank}...")
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ConvUnetExtractor(
        num_layers=12,
        embed_dim=560,
        attention_heads=20,
        alphabet=alphabet
    )
    state = torch.load(args.model_path, map_location="cpu")
    clean_state = OrderedDict(
        (k.replace("_orig_mod.", "").replace("module.", ""), v)
        for k, v in state.items()
    )
    model.load_state_dict(clean_state, strict=False)
    model.to(device)

    model.eval()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    records = read_fasta(args.input_fasta, args.pid_form)
    dataset = FastaDataset(records)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, alphabet),
        num_workers=4,
        pin_memory=True
    )
    if local_rank == 0:
        print(f"{len(records)} sequences loaded")

    local_results = {}
    with torch.no_grad():
        for labels, seqs, batch_tokens, seq_lengths in tqdm(loader, disable=(local_rank != 0)):
            batch_tokens = batch_tokens.to(device)
            outputs = model(batch_tokens)
            outputs = outputs.cpu().numpy()
            for i, (sid, seq) in enumerate(zip(labels, seqs)):
                seq_len = seq_lengths[i]
                emb = outputs[i, :seq_len, :].astype(np.float32)
                local_results[sid] = emb

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_results)

    if local_rank == 0:
        merged = {}
        for x in gathered:
            merged.update(x)
        print(f"Writing {len(merged)} embeddings to {args.output_h5}...")
        with h5py.File(args.output_h5, "w") as h5f:
            for sid, emb in tqdm(merged.items()):
                grp = h5f.create_group(sid)
                grp.create_dataset("embedding", data=emb, compression="gzip")
        print("DONE")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
