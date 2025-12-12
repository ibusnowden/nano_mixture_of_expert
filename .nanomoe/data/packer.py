#packed memmap
# nanomoe/data/packed_memmap.py
import numpy as np
import torch
from torch.utils.data import Dataset

class PackedMemMapDataset(Dataset):
    """
    Expects a memmap of int32 token IDs already packed into fixed-length sequences.
    Shape: [N, S]
    """
    def __init__(self, path: str, seq_len: int):
        self.seq_len = seq_len
        self.arr = np.memmap(path, mode="r", dtype=np.int32)
        assert self.arr.size % seq_len == 0
        self.n = self.arr.size // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.arr[i*self.seq_len:(i+1)*self.seq_len].astype(np.int64)
        x = torch.from_numpy(x)
        return x
