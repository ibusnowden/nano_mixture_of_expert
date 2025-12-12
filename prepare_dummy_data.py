import os
import numpy as np
from nanomoe.configs.base import TrainConfig

def create_dummy_data(out_path="data_train.memmap", size_mb=10, vocab_size=32000):
    """
    Creates a dummy memmap file with random tokens.
    """
    print(f"Generating dummy data at {out_path}...")
    # Calculate number of int32 tokens
    # 1 MB = 1024 * 1024 bytes
    # int32 = 4 bytes
    n_tokens = (size_mb * 1024 * 1024) // 4
    
    # Generate random tokens
    arr = np.random.randint(0, vocab_size, size=n_tokens, dtype=np.int32)
    
    # Write to memmap
    mm = np.memmap(out_path, mode="w+", dtype=np.int32, shape=(n_tokens,))
    mm[:] = arr[:]
    mm.flush()
    print(f"Done. Created {out_path} with {n_tokens} tokens ({size_mb} MB).")

if __name__ == "__main__":
    cfg = TrainConfig()
    create_dummy_data(vocab_size=cfg.vocab_size)
