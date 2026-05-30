import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import esm
import sys
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relic.modules import (clean_state_dict,
                              top_k_top_p_filtering,
                              apply_repetition_penalty,
                              calc_banned_ngram_tokens)
from relic.gpt import ProteinGPT
from relic.encoder import HybridEncoder
from relic.vq import (ProteinCodeDecoder,
                         TransformerAdapter,
                         TransformerDecoderAdapter,
                         ResidualVectorQuantizer,
                         HybridDecoder)

alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

def load_vq_model(vq_ckpt, device, alphabet):
    print(f"Loading VQ Checkpoint: {vq_ckpt}")
    vq_layer = ResidualVectorQuantizer(8192, 384, 1)
    decoder_adapter = TransformerDecoderAdapter(384, 560, num_heads=20, num_layers=16)
    decoder = HybridDecoder(560, len(alphabet), num_heads=20)
    model = ProteinCodeDecoder(vq_layer, decoder_adapter, decoder).to(device)
    vq_sd = torch.load(vq_ckpt, map_location='cpu')
    comps = {
        'vq': model.codebook,
        'decoder_adapter': model.decoder_adapter,
        'decoder': model.decoder
    }
    for key, sub_sd in vq_sd.items():
        if key in comps:
            target_module = comps[key]
            if key == 'vq':
                clean_sub = OrderedDict()
                found_weight = False
                for k, v in sub_sd.items():
                    clean_k = k.replace("module.", "").replace("_orig_mod.", "")
                    if "embedding.weight" in clean_k:
                        clean_sub["weight"] = v
                        found_weight = True
                        break
                if found_weight:
                    target_module.load_state_dict(clean_sub, strict=True)
            else:
                clean_sub = OrderedDict()
                for k, v in sub_sd.items():
                    clean_k = k.replace("module.", "").replace("_orig_mod.", "")
                    clean_sub[clean_k] = v
                target_module.load_state_dict(clean_sub, strict=True)
    model.eval()
    return model

class ProteinGPTConfig:
    def __init__(self, code_vocab_size=8192, embed_dim=1024, layers=30, heads=16, dropout=0.0):
        self.code_vocab_size = code_vocab_size
        self.embed_dim = embed_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout

@torch.no_grad()
def generate_batch(gpt_model, vq_model, gpt_codebook, alphabet, args, batch_size):
    device = args.device
    gpt_model.eval()
    vq_model.eval()

    sos_idx = args.code_vocab_size + 1
    eos_idx = args.code_vocab_size + 2

    seq_ids = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

    normed_codebook = None
    if args.similarity_threshold > -1:
        normed_codebook = F.normalize(gpt_codebook, p=2, dim=1)

    for step in range(args.max_len):
        B, L = seq_ids.shape
        pos_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        attn_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        attn_mask_float = torch.zeros((L, L), device=device, dtype=torch.bfloat16)
        attn_mask_float.masked_fill_(attn_mask, float('-inf'))
        attn_mask_float = attn_mask_float.view(1, 1, L, L).expand(B, 1, L, L)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = gpt_model(seq_ids, pos_ids, attn_mask_float)

        next_token_logits = logits[:, -1, :].clone()
        
        if normed_codebook is not None:
            for b in range(B):
                anchor_idx = torch.argmax(next_token_logits[b])
                if anchor_idx < args.code_vocab_size:
                    anchor_emb = normed_codebook[anchor_idx].unsqueeze(0)
                    sims = F.cosine_similarity(anchor_emb, normed_codebook)
                    semantic_mask = sims < args.similarity_threshold
                    semantic_mask[anchor_idx] = False
                    current_logits = next_token_logits[b, :args.code_vocab_size]
                    current_logits[semantic_mask] = -float('inf')
                    next_token_logits[b, :args.code_vocab_size] = current_logits
                    
        if args.no_repeat_ngram_size > 0:
            banned = calc_banned_ngram_tokens(seq_ids, B, args.no_repeat_ngram_size, L)
            for batch_idx, banned_tokens in enumerate(banned):
                for tok in banned_tokens:
                    next_token_logits[batch_idx, tok] = -float("inf")
                    
        if args.rep_pen != 1.0:
            next_token_logits = apply_repetition_penalty(next_token_logits, seq_ids[:, 1:], args.rep_pen)
            
        next_token_logits = next_token_logits / args.temperature
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        next_token[~unfinished_sequences] = eos_idx
        seq_ids = torch.cat([seq_ids, next_token], dim=1)
        
        unfinished_sequences = unfinished_sequences & (next_token.squeeze(-1) != eos_idx)
        if not unfinished_sequences.any():
            break


    results = []
    special_tokens = {alphabet.cls_idx, alphabet.padding_idx, alphabet.eos_idx, alphabet.unk_idx, alphabet.mask_idx}
    
    for b in range(batch_size):
        seq = seq_ids[b, 1:]
        
        eos_positions = (seq == eos_idx).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            seq = seq[:eos_positions[0]]
            
        if len(seq) == 0:
            results.append("")
            continue
            
        indices = seq.unsqueeze(0).to(device)
        quantized = F.embedding(indices, gpt_codebook).permute(0, 2, 1)
        restored = vq_model.decoder_adapter(quantized)
        
        L_feat = restored.shape[2]
        mask_l4 = torch.zeros((1, L_feat), dtype=torch.bool, device=device)
        mask_l2 = torch.zeros((1, L_feat * 2), dtype=torch.bool, device=device)
        
        dec_logits = vq_model.decoder(restored, mask_l4=mask_l4, mask_l2=mask_l2)
        pred_tokens = dec_logits.argmax(dim=-1).squeeze(0)
        
        decoded_aa = "".join([alphabet.get_tok(t) for t in pred_tokens.cpu().numpy() if t not in special_tokens])
        results.append(decoded_aa)
        
    return results

def worker(rank, num_gpus, args):
    device = torch.device(f"cuda:{rank}")
    args.device = device
    
    samples_per_gpu = args.num_samples // num_gpus
    if rank < args.num_samples % num_gpus:
        samples_per_gpu += 1
        
    if samples_per_gpu == 0:
        return []

    gpt_config = ProteinGPTConfig(
        code_vocab_size=args.code_vocab_size,
        embed_dim=args.embed_dim, layers=args.layers, heads=args.heads
    )
    gpt_model = ProteinGPT(gpt_config).to(device)

    gpt_ckpt = torch.load(args.gpt_ckpt, map_location='cpu')
    if 'state_dict' in gpt_ckpt: gpt_ckpt = gpt_ckpt['state_dict']
    clean_gpt_sd = clean_state_dict(gpt_ckpt)
    gpt_model.load_state_dict(clean_gpt_sd, strict=False)

    vq_model = load_vq_model(args.vq_ckpt, device, alphabet)
    gpt_codebook = vq_model.codebook.weight.data.to(device)

    all_res = []
    num_batches = (samples_per_gpu + args.batch_size - 1) // args.batch_size
    
    for _ in tqdm(range(num_batches), desc=f"GPU {rank}", position=rank):
        current_batch = min(args.batch_size, samples_per_gpu - len(all_res))
        if current_batch <= 0: break
        batch_seqs = generate_batch(gpt_model, vq_model, gpt_codebook, alphabet, args, current_batch)
        all_res.extend(batch_seqs)
        
    return all_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Protein Sequences (Multi-GPU Batch)")

    parser.add_argument("--gpt_ckpt", type=str, required=True)
    parser.add_argument("--vq_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated.fasta")

    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rep_pen", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--similarity_threshold", type=float, default=-1.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--code_vocab_size", type=int, default=8192)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--vq_adapter_dim", type=int, default=384)
    parser.add_argument("--vq_decoder_dim", type=int, default=560)

    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit(1)

    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(worker, [(rank, num_gpus, args) for rank in range(num_gpus)])

    flat_results = [seq for sublist in results for seq in sublist]
    
    with open(args.output, 'w') as f:
        for idx, seq in enumerate(flat_results):
            f.write(f">Gen_{idx}\n")
            f.write(f"{seq}\n")
