import json
import numpy as np
from typing import Iterable
from nanomoe.data.tokenizer_tiktoken import TikTokenizer

def iter_jsonl_texts(path: str, field: str = "text") -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row[field]

def pack_texts_to_memmap(texts: Iterable[str], out_path: str, seq_len: int, tok: TikTokenizer):
    flat = []
    for t in texts:
        flat.extend(tok.encode(t))

    arr = np.asarray(flat, dtype=np.int32)
    n = (arr.size // seq_len) * seq_len
    arr = arr[:n]
    if n == 0:
        raise ValueError("No tokens packed; check input.")

    mm = np.memmap(out_path, mode="w+", dtype=np.int32, shape=(n,))
    mm[:] = arr
    mm.flush()
    print(f"[pack] {n} tokens -> {n//seq_len} sequences -> {out_path}")
