from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)

        self.merge_ranks = dict[tuple[bytes, bytes], int]()

        for i, p in enumerate(merges):
            self.merge_ranks[p] = i

        if special_tokens is not None:
            self.special_tokens = list(special_tokens)
            for token in special_tokens:
                if token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token.encode("utf-8")
        else:
            self.special_tokens = []

        self.reverse_vocab = {v: k for k,v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: str | None = None) -> "Tokenizer":
        return
    
    def encode(self, text: str) -> list[int]:
        return
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return

    def decode(self, ids: list[int]) -> str:
        return
