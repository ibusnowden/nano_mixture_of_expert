# nanomoe/data/quality/rephrase_verify_pack.py
"""
v0.1 pipeline (offline):
  1) Load seed items (jsonl): {id, prompt, reference_solution?, tests?, metadata}
  2) Rephrase prompts using gpt-oss 20B (local or server)
  3) (Optional) Generate candidate solutions with teacher(s)
  4) Verify: unit tests / compilation / simple heuristics
  5) Pack accepted samples into fixed-length token sequences and write memmap

This file intentionally separates:
  - model calls (rephrase/generate)
  - verification
  - packing
so you can swap components without touching training.
"""
import os, json, random
import numpy as np

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Rephrasing (stub) ----------
def rephrase_with_gptoss20b(prompt: str, n: int) -> list[str]:
    """
    Replace with:
      - local vLLM server call
      - or HF generate
    Must enforce diversity styles to avoid “single-voice” overfit.
    """
    # v0.1 placeholder: simple templates
    styles = [
        lambda p: f"Rewrite the task precisely and tersely:\n{p}",
        lambda p: f"Restate with explicit constraints and shapes:\n{p}",
        lambda p: f"Restate like a StackOverflow question:\n{p}",
        lambda p: f"Restate adversarially (remove hints):\n{p}",
    ]
    out = []
    for i in range(n):
        out.append(styles[i % len(styles)](prompt))
    return out

# ---------- Verification (stubs) ----------
def verify_code_solution(sample: dict) -> tuple[bool, str]:
    """
    Ideally offers multiple verification tiers:
      Tier0: cheap heuristics (length, forbidden tokens, obvious nonsense)
      Tier1: compile / import
      Tier2: unit tests + numerical checks
      Tier3: perf sanity (optional)
    """
    # v0.1: accept all
    return True, "ok"

# ---------- Tokenization/Packing ----------
def tokenize(text: str) -> np.ndarray:
    """
    Replace with your tokenizer.
    v0.1: fake tokenizer by bytes (NOT for real training).
    """
    b = text.encode("utf-8")[:4096]
    return np.frombuffer(b, dtype=np.uint8).astype(np.int32)

def pack_to_memmap(texts: list[str], seq_len: int, out_path: str):
    # concatenate tokens with EOS=0 (placeholder)
    eos = np.array([0], dtype=np.int32)
    toks = []
    for t in texts:
        toks.append(tokenize(t))
        toks.append(eos)
    flat = np.concatenate(toks, axis=0)

    # trim to multiple of seq_len
    n = (flat.size // seq_len) * seq_len
    flat = flat[:n]

    mm = np.memmap(out_path, mode="w+", dtype=np.int32, shape=(n,))
    mm[:] = flat[:]
    mm.flush()

def main():
    seed_path = os.environ.get("SEED_JSONL", "seeds.jsonl")
    out_jsonl = os.environ.get("OUT_JSONL", "accepted.jsonl")
    out_memmap = os.environ.get("OUT_MEMMAP", "data_train.memmap")
    seq_len = int(os.environ.get("SEQ_LEN", "2048"))
    n_rephrase = int(os.environ.get("N_REPHRASE", "4"))

    accepted = []
    packed_texts = []

    for row in load_jsonl(seed_path):
        prompt = row["prompt"]
        variants = [prompt] + rephrase_with_gptoss20b(prompt, n=n_rephrase)

        for v in variants:
            sample = dict(row)
            sample["prompt_variant"] = v

            ok, reason = verify_code_solution(sample)
            if not ok:
                continue

            sample["verify_reason"] = reason
            accepted.append(sample)

            # pack prompt + (optional) solution into a single training text
            sol = sample.get("solution", "")
            packed_texts.append(f"<|prompt|>\n{v}\n<|solution|>\n{sol}\n")

    save_jsonl(out_jsonl, accepted)
    pack_to_memmap(packed_texts, seq_len=seq_len, out_path=out_memmap)
    print(f"accepted={len(accepted)} memmap={out_memmap}")

if __name__ == "__main__":
    main()
