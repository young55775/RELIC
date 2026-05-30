import os
import pickle as pkl
import random
import torch
from torch.utils.data import Dataset
import esm


class IndexedUniRef(Dataset):
    """ Used for Stage 1 (MLM) and Stage 2 (VQ) """

    def __init__(self, fasta_file, index_file, target_len):
        self.target_len = target_len
        self.fasta_file = fasta_file
        with open(index_file, 'rb') as f:
            self.index = pkl.load(f)
        self.f = None

    def _open_file(self):
        if self.f is None: self.f = open(self.fasta_file, 'r')

    def __getitem__(self, idx):
        offset = self.index[idx]
        try:
            self._open_file()
            self.f.seek(offset)
            header = self.f.readline().strip()
            if not header.startswith(">"): raise ValueError("Bad Offset")
            seq_id = header[1:].split()[0]
            parts = []
            while True:
                line = self.f.readline()
                if not line or line.startswith('>'): break
                parts.append(line.strip())
            seq = "".join(parts).replace('j', 'i').replace('J', 'I')
        except:
            return "ERROR", "X", False, False

        # Random Cropping Logic
        original_len = len(seq)
        if original_len > self.target_len:
            start = random.randint(0, original_len - self.target_len)
            processed = seq[start: start + self.target_len]
            return seq_id, processed, (start == 0), (start + self.target_len == original_len)
        else:
            # Short sequence logic (simplified for brevity)
            return seq_id, seq, True, True

    def __len__(self):
        return len(self.index)


class GPTDataset(Dataset):
    """ Used for Stage 3 (Generative) """

    def __init__(self, fpath, go_vocab):
        self.data = []
        with open(fpath, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines): break
                header = lines[i].strip()
                seq_codes = [int(x) for x in lines[i + 1].strip().split()]
                gos = []
                if header.startswith(">"):
                    for p in header[1:].split('|'):
                        if p: gos.append(go_vocab.get(p, 1))
                self.data.append((gos, seq_codes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
