import torch
import torch.nn.functional as F
import numpy as np
import math

def coords_to_contact_map(coords: np.ndarray, threshold: float = 8.0) -> torch.Tensor:

    # delta shape: (L, L, 3)
    delta = coords[:, None, :] - coords[None, :, :]
    # dist shape: (L, L)
    dist = np.sqrt((delta ** 2).sum(-1))

    contact = (dist < threshold).astype(np.float32)
    return torch.from_numpy(contact)


def load_npz_data(npz_path, target_len=None):
    data = np.load(npz_path)
    seq = str(data['seq'])
    coords = data['coords']

    if target_len and len(seq) > target_len:
        import random
        start = random.randint(0, len(seq) - target_len)
        seq = seq[start: start + target_len]
        coords = coords[start: start + target_len]

    contact_map = coords_to_contact_map(coords)
    return seq, contact_map

def pad_for_static_inference(token_ids: torch.Tensor, static_len: int, pad_idx: int = 1) -> torch.Tensor:

    L = len(token_ids)
    if L > static_len:
        return token_ids[:static_len]

    padding_needed = static_len - L
    return F.pad(token_ids, (0, padding_needed), value=pad_idx)


def batch_static_collate(batch_tokens, static_len=1500, pad_idx=1):
    padded_batch = []
    lengths = []
    for tokens in batch_tokens:
        lengths.append(len(tokens))
        padded_batch.append(pad_for_static_inference(tokens, static_len, pad_idx))
    return torch.stack(padded_batch), torch.tensor(lengths)
