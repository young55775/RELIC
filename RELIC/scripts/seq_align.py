import argparse
import math
import os
import random

import esm
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import TransformerLayer, ESM1bLayerNorm, RobertaLMHead
from numba import jit
from sklearn.neighbors import NearestNeighbors

# SW Params
GAP_OPEN = 5.0
GAP_EXTEND = 1.0
SIM_THRESHOLD = 1.0
DECAY_SIGMA = 100.0

# Search Params
TOKEN_NEIGHBORS = 1000
CANDIDATE_LIMIT = 10000
EMBEDDING_DIM = 560

# Stats & Output
N_RANDOM_PAIRS_FOR_STATS = 10000
Z_SCORE_THRESHOLD = -99.0
TOP_K_RESULTS = 50


# ================= Numba Kernels =================
@jit(nopython=True, fastmath=True)
def apply_diagonal_decay(sim_matrix, sigma):
    rows, cols = sim_matrix.shape
    decayed_matrix = np.empty((rows, cols), dtype=np.float32)
    denom = 2 * sigma * sigma
    for i in range(rows):
        rel_i = i / (rows + 1e-6)
        for j in range(cols):
            rel_j = j / (cols + 1e-6)
            diff = rel_i - rel_j
            weight = np.exp(-(diff * diff) / denom)
            decayed_matrix[i, j] = sim_matrix[i, j] * weight
    return decayed_matrix


@jit(nopython=True, fastmath=True)
def shifted_quadratic_sw_score_only(sim_matrix, gap_open, _gap_ext, threshold):
    rows, cols = sim_matrix.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    max_score = 0.0
    NOISE_PENALTY = 0.1

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            sim = sim_matrix[i - 1, j - 1]
            if sim > threshold:
                match_score = sim * sim
            else:
                match_score = -NOISE_PENALTY

            score_diag = H[i - 1, j - 1] + match_score
            score_up = H[i - 1, j] - gap_open
            score_left = H[i, j - 1] - gap_open

            current_score = max(0.0, score_diag, score_up, score_left)
            H[i, j] = current_score
            if current_score > max_score:
                max_score = current_score
    return max_score


@jit(nopython=True, fastmath=True)
def shifted_quadratic_sw_traceback_detailed(sim_matrix, gap_open, _gap_ext, threshold):
    rows, cols = sim_matrix.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    Dir = np.zeros((rows + 1, cols + 1), dtype=np.uint8)  # 0=Stop, 1=Diag, 2=Up, 3=Left

    max_score = 0.0
    end_i, end_j = 0, 0
    NOISE_PENALTY = 0.1

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            sim = sim_matrix[i - 1, j - 1]
            if sim > threshold:
                match_score = sim * sim
            else:
                match_score = -NOISE_PENALTY

            score_diag = H[i - 1, j - 1] + match_score
            score_up = H[i - 1, j] - gap_open
            score_left = H[i, j - 1] - gap_open

            current_score = 0.0
            direction = 0

            # Priority: Diag > Up > Left
            if score_diag >= current_score:
                current_score = score_diag
                direction = 1
            if score_up > current_score:
                current_score = score_up
                direction = 2
            if score_left > current_score:
                current_score = score_left
                direction = 3

            H[i, j] = current_score
            Dir[i, j] = direction

            if current_score > max_score:
                max_score = current_score
                end_i, end_j = i, j

    n_matched = 0
    n_mismatched = 0
    n_gap_moves = 0

    q_aligned_residues = 0
    t_aligned_residues = 0

    curr_i, curr_j = end_i, end_j

    while curr_i > 0 and curr_j > 0 and H[curr_i, curr_j] > 0:
        d = Dir[curr_i, curr_j]

        if d == 1:  # Diag: Match or Mismatch
            sim = sim_matrix[curr_i - 1, curr_j - 1]
            if sim > threshold:
                n_matched += 1
            else:
                n_mismatched += 1

            q_aligned_residues += 1
            t_aligned_residues += 1

            curr_i -= 1
            curr_j -= 1

        elif d == 2:  # Up: Gap in Target (Target index j stays same, Query index i moves)
            n_gap_moves += 1
            q_aligned_residues += 1
            curr_i -= 1

        elif d == 3:  # Left: Gap in Query (Query index i stays same, Target index j moves)
            n_gap_moves += 1
            t_aligned_residues += 1
            curr_j -= 1

        else:
            break

    return max_score, n_matched, n_mismatched, n_gap_moves, q_aligned_residues, t_aligned_residues


@jit(nopython=True, fastmath=True)
def get_sw_path_for_plot(sim_matrix, gap_open, _gap_ext, threshold):
    rows, cols = sim_matrix.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    Dir = np.zeros((rows + 1, cols + 1), dtype=np.uint8)

    max_score = 0.0
    end_i, end_j = 0, 0
    NOISE_PENALTY = 0.1

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            sim = sim_matrix[i - 1, j - 1]
            match_score = (sim * sim) if sim > threshold else -NOISE_PENALTY
            score_diag = H[i - 1, j - 1] + match_score
            score_up = H[i - 1, j] - gap_open
            score_left = H[i, j - 1] - gap_open

            current_score = 0.0
            direction = 0
            if score_diag >= current_score: current_score, direction = score_diag, 1
            if score_up > current_score: current_score, direction = score_up, 2
            if score_left > current_score: current_score, direction = score_left, 3

            H[i, j] = current_score
            Dir[i, j] = direction
            if current_score > max_score: max_score, end_i, end_j = current_score, i, j

    path_i = []
    path_j = []
    curr_i, curr_j = end_i, end_j

    while curr_i > 0 and curr_j > 0 and H[curr_i, curr_j] > 0:
        d = Dir[curr_i, curr_j]
        path_i.append(curr_i - 1)
        path_j.append(curr_j - 1)

        if d == 1:
            curr_i -= 1
            curr_j -= 1
        elif d == 2:
            curr_i -= 1
        elif d == 3:
            curr_j -= 1
        else:
            break

    return path_i, path_j


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
            pooled_mask = torch.nn.functional.max_pool1d(torch.nn.functional.max_pool1d(m, 2, 2), 2, 2).bool().squeeze(
                1)

        attn_out, _ = self.bottleneck_transformer(attn_in, self_attn_padding_mask=pooled_mask)
        if return_bottleneck: return attn_out.transpose(0, 1)

        h_upsample_in = attn_out.permute(1, 2, 0)
        h_up1 = self.activation(self._apply_norm(self.norm_t1, self.tconv1(self.unpool1(h_upsample_in) + skip2)))
        h_up2 = self.activation(self._apply_norm(self.norm_t2, self.tconv2(self.unpool2(h_up1) + skip1)))
        output = x + self.dropout_layer(h_up2.permute(2, 0, 1))
        return output


class ConvunetExtractor(nn.Module):
    def __init__(self, num_layers=12, embed_dim=560, attention_heads=20, alphabet="ESM-1b"):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        if not isinstance(alphabet, esm.data.Alphabet): alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.embed_tokens = nn.Embedding(len(alphabet), embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            ConvUnetAttentionLayer(embed_dim, 4 * embed_dim, attention_heads) for _ in range(num_layers)
        ])
        self.emb_layer_norm_after = ESM1bLayerNorm(embed_dim)
        self.lm_head = RobertaLMHead(embed_dim, len(alphabet), self.embed_tokens.weight)

    def forward(self, tokens):
        padding_mask = tokens.eq(self.padding_idx)
        x = self.embed_tokens(tokens)
        if padding_mask is not None: x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        if not padding_mask.any(): padding_mask = None
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                return layer(x, self_attn_padding_mask=padding_mask, return_bottleneck=True)
            else:
                x = layer(x, self_attn_padding_mask=padding_mask, return_bottleneck=False)
        return x


# ================= Core computing logic =================
def transform_data(X, mu, W):
    if X.shape[0] == 0: return X
    if X.ndim == 1:
        X = X.reshape(1, -1)
        return np.dot(X - mu, W).astype(np.float32).flatten()
    return np.dot(X - mu, W).astype(np.float32)


def calculate_sw_score_simple(Q_emb, T_emb):
    sim_matrix = np.dot(Q_emb, T_emb.T)
    if DECAY_SIGMA < 10.0:
        decayed_matrix = apply_diagonal_decay(sim_matrix, DECAY_SIGMA)
    else:
        decayed_matrix = sim_matrix

    raw_score = shifted_quadratic_sw_score_only(decayed_matrix, GAP_OPEN, GAP_EXTEND, SIM_THRESHOLD)
    denom = np.sqrt(Q_emb.shape[0] * T_emb.shape[0]) + 1e-6
    return raw_score / denom


def calculate_sw_score_detailed(Q_emb, T_emb):
    sim_matrix = np.dot(Q_emb, T_emb.T)
    if DECAY_SIGMA < 10.0:
        decayed_matrix = apply_diagonal_decay(sim_matrix, DECAY_SIGMA)
    else:
        decayed_matrix = sim_matrix

    # Detailed Traceback with Precision Counts
    raw_score, n_match, n_mismatch, n_gaps, q_aligned_res, t_aligned_res = shifted_quadratic_sw_traceback_detailed(
        decayed_matrix, GAP_OPEN, GAP_EXTEND, SIM_THRESHOLD
    )

    denom = np.sqrt(Q_emb.shape[0] * T_emb.shape[0]) + 1e-6
    final_score = raw_score / denom

    return {
        'score': final_score,
        'raw': raw_score,
        'n_matched': n_match,
        'n_mismatched': n_mismatch,
        'n_gaps': n_gaps,
        'q_aligned_len': q_aligned_res,
        't_aligned_len': t_aligned_res
    }


# ================= Stats =================
def check_and_load_stats(h5_path):
    db_name = os.path.basename(h5_path)
    cache_path = os.path.join(os.path.dirname(h5_path), f"{db_name}.stats_v16_sw_gap5.npz")

    if os.path.exists(cache_path):
        print(f"[Cache] Loading stats from {cache_path}...")
        try:
            data = np.load(cache_path)
            return data['mu_global'], data['W_global'], float(data['mu_score']), float(data['sigma_score'])
        except:
            pass

    print("[Stats] Calculating stats (SW Logic)...")
    mu_global = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    N_total = 0
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"Pass 1: {len(keys)} keys")
        for key in keys:  # Pass 1
            if 'embedding' in f[key]:
                vec = f[key]['embedding'][:]
                if vec.size > 0:
                    mu_global += np.sum(vec, axis=0)
                    N_total += vec.shape[0]
    mu_global /= N_total

    cov_global = np.zeros((EMBEDDING_DIM, EMBEDDING_DIM), dtype=np.float64)
    with h5py.File(h5_path, 'r') as f:
        print(f"Pass 2: {len(keys)} keys")
        for key in keys:  # Pass 2
            vec = f[key]['embedding'][:]
            X_centered = vec - mu_global
            cov_global += np.dot(X_centered.T, X_centered)
    cov_global /= (N_total - 1)
    U, S, _ = np.linalg.svd(cov_global)
    W_global = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-5)))

    print(f"Sampling {N_RANDOM_PAIRS_FOR_STATS} pairs for SW Z-Stats...")
    bg_scores = []
    pairs = [random.sample(keys, 2) for _ in range(N_RANDOM_PAIRS_FOR_STATS)]
    mu_f32 = mu_global.astype(np.float32)
    W_f32 = W_global.astype(np.float32)

    with h5py.File(h5_path, 'r') as f:
        for pid1, pid2 in pairs:
            v1 = transform_data(f[pid1]['embedding'][:], mu_f32, W_f32)
            v2 = transform_data(f[pid2]['embedding'][:], mu_f32, W_f32)
            if v1.shape[0] > 0 and v2.shape[0] > 0:
                s = calculate_sw_score_simple(v1, v2)
                bg_scores.append(s)

    mu_score = np.mean(bg_scores)
    sigma_score = np.std(bg_scores)

    print(f"   -> Mu_BG: {mu_score:.4f}, Sigma_BG: {sigma_score:.4f}")
    np.savez(cache_path, mu_global=mu_f32, W_global=W_f32, mu_score=mu_score, sigma_score=sigma_score)
    return mu_f32, W_f32, mu_score, sigma_score


# ================= KNN Loader =================
def load_db_into_memory(h5_path, mu_global, W_global):
    print(f"[DB] Loading FULL database for KNN...")
    pids = []
    vectors = []
    token_owner = []
    offsets = []
    curr = 0
    mu_f32 = mu_global.astype(np.float32)
    W_f32 = W_global.astype(np.float32)

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"Loading DB: {len(keys)} keys")
        for i, pid in enumerate(keys):
            if 'embedding' not in f[pid]:
                offsets.append((curr, 0))
                pids.append(pid)
                continue
            vec = f[pid]['embedding'][:]
            if vec.shape[0] == 0:
                offsets.append((curr, 0))
                pids.append(pid)
                continue

            vec_white = transform_data(vec, mu_f32, W_f32)
            vectors.append(vec_white)
            token_owner.extend([i] * vec_white.shape[0])
            l = vec_white.shape[0]
            offsets.append((curr, l))
            curr += l
            pids.append(pid)

    GLOBAL_DB_NP = np.vstack(vectors).astype(np.float32)
    GLOBAL_TOKEN_OWNER = np.array(token_owner, dtype=np.int32)
    return GLOBAL_DB_NP, GLOBAL_TOKEN_OWNER, offsets, pids


# ================= Main =================
def get_query_embedding(model, alphabet, sequence, device):
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("query", sequence)])
    batch_tokens = batch_tokens.to(device)
    seq_len = batch_tokens.shape[1]
    target_len = math.ceil(seq_len / 4) * 4
    if target_len - seq_len > 0:
        batch_tokens = F.pad(batch_tokens, (0, target_len - seq_len), value=alphabet.padding_idx)

    with torch.no_grad():
        results = model(batch_tokens)

    token_embeddings = results[:, :seq_len, :][0, :, :].cpu().numpy()
    return token_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq", required=True, help="Sequence to align")
    parser.add_argument("-p", "--param_path", required=True, help="Parameter file")
    parser.add_argument("-e", "--embed_path", required=True, help="Embedding file of genome")
    parser.add_argument("-o", "--output_csv", default="search_results.csv",
                        help="Align results file path (default: search_results.csv)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--force_brute", action="store_true", help="Skip KNN")
    args = parser.parse_args()

    # 1. Load Stats
    mu_global, W_global, mu_score, sigma_score = check_and_load_stats(args.embed_path)

    # 2. Query
    print(f"Query length: {len(args.seq)} (+2 tokens)")
    try:
        query_raw_emb = get_query_embedding(model, alphabet, args.seq, args.device)
    except Exception as e:
        print(f"[Error] Inference error: {e}")
        return

    query_emb = transform_data(query_raw_emb, mu_global, W_global)
    q_len = query_emb.shape[0]

    (GLOBAL_DB_NP,
     GLOBAL_TOKEN_OWNER,
     GLOBAL_OFFSETS,
     GLOBAL_PIDS) = load_db_into_memory(args.token_path, mu_global, W_global)
    num_seqs = len(GLOBAL_PIDS)

    if args.force_brute:
        print("[Mode] FORCE BRUTE.")
        target_indices = list(range(num_seqs))
    else:
        print(f"KNN Voting (K={TOKEN_NEIGHBORS})...")
        knn = NearestNeighbors(n_neighbors=TOKEN_NEIGHBORS, metric='cosine', algorithm='brute', n_jobs=-1)
        knn.fit(GLOBAL_DB_NP)
        _, indices = knn.kneighbors(query_emb)
        token_hits = indices.flatten()
        neighbor_pids = GLOBAL_TOKEN_OWNER[token_hits]
        counts = np.bincount(neighbor_pids, minlength=num_seqs)

        if len(counts) > CANDIDATE_LIMIT:
            cand_indices = np.argpartition(counts, -CANDIDATE_LIMIT)[-CANDIDATE_LIMIT:]
            target_indices = [idx for idx in cand_indices if counts[idx] > 0]
        else:
            target_indices = np.where(counts > 0)[0]
        print(f"   -> Selected {len(target_indices)} candidates.")

    print(f"SW Scoring & Detailed Traceback...")
    results_list = []

    # Warmup Numba
    _ = shifted_quadratic_sw_score_only(np.zeros((10, 10), dtype=np.float32), 5.0, 1.0, 1.0)

    print(f"Refining: {len(target_indices)} indices")
    for t_idx in target_indices:  # Refining
        t_start, t_len = GLOBAL_OFFSETS[t_idx]
        if t_len == 0: continue

        target_emb = GLOBAL_DB_NP[t_start: t_start + t_len]

        # Detailed with Exact Lengths
        metrics = calculate_sw_score_detailed(query_emb, target_emb)

        if sigma_score > 1e-6:
            z_score = (metrics['score'] - mu_score) / sigma_score
        else:
            z_score = metrics['score']

        if z_score > Z_SCORE_THRESHOLD:
            results_list.append({
                'Protein': GLOBAL_PIDS[t_idx],
                'Z-Score': z_score,
                'Raw_Score': metrics['raw'],
                'N_Match': metrics['n_matched'],
                'N_Mismatch': metrics['n_mismatched'],
                'N_Gap': metrics['n_gaps'],
                'Q_Cov': metrics['q_aligned_len'] / q_len,
                'T_Cov': metrics['t_aligned_len'] / t_len,
                'Identity': metrics['n_matched'] / (metrics['q_aligned_len'] + 1e-6),
                'Len_Q': q_len,
                'Len_T': t_len
            })

    if not results_list:
        print(f"[Error] No results found.")
        return

    df = pd.DataFrame(results_list)
    df = df.sort_values(by='Z-Score', ascending=False).head(TOP_K_RESULTS).reset_index(drop=True)

    cols = ['Protein', 'Z-Score', 'Q_Cov', 'T_Cov', 'Identity', 'N_Match', 'N_Mismatch', 'N_Gap', 'Len_Q', 'Len_T',
            'Raw_Score']
    df = df[cols]
    df['Z-Score'] = df['Z-Score'].map('{:,.2f}'.format)
    df['Q_Cov'] = df['Q_Cov'].map('{:.1%}'.format)
    df['T_Cov'] = df['T_Cov'].map('{:.1%}'.format)
    df['Identity'] = df['Identity'].map('{:.1%}'.format)
    df['Raw_Score'] = df['Raw_Score'].map('{:.3e}'.format)

    df.to_csv(args.output_csv, index=False)
    print(f"Done {TOP_K_RESULTS} Search Results")


if __name__ == "__main__":
    main()
