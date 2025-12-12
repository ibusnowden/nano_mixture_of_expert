from dataclasses import dataclass
from typing import List
import tiktoken

@dataclass(frozen=True)
class TikTokenizer:
    """
    We reserve token id 0 as EOS/document boundary.
    We shift all tiktoken ids by +1.
    """
    name: str = "cl100k_base"
    add_eos: bool = True
    eos_id: int = 0

    def __post_init__(self):
        enc = tiktoken.get_encoding(self.name)
        object.__setattr__(self, "_enc", enc)
        object.__setattr__(self, "vocab_size", enc.n_vocab + 1)

    def encode(self, text: str) -> List[int]:
        ids = self._enc.encode_ordinary(text)
        ids = [i + 1 for i in ids]
        if self.add_eos:
            ids.append(self.eos_id)
        return ids
