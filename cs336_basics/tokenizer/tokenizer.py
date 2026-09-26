from collections.abc import Iterable, Iterator
from .serialization import load_gpt2_vocab, load_gpt2_merges
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    # looks like the vocab file is actually in the form of reversed vocab
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab = load_gpt2_vocab(vocab_filepath)
        merges = load_gpt2_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    # applies the merges to a pre-token word
    def _apply_merges_word(self, word: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # go down the word, look at every adjacent pair, merge the one that's the lowest rank in merge_ranks
        merge_word = word
        while len(merge_word) > 1:
            # get a list of the bytes pairs in the word
            temp_pairs = []
            for index in range(len(merge_word) - 1):
                temp_pairs.append((merge_word[index], merge_word[index + 1]))
            # find the one that has the lowest merge rank if it exists
            min_rank = float("inf")
            best_pair = []
            for pair in temp_pairs:
                if pair in self.merge_ranks:
                    if self.merge_ranks.get(pair) < min_rank:
                        min_rank = self.merge_ranks.get(pair)
                        best_pair = pair
            # if a pair never is in self.merge_ranks, then min_rank is still larger than possible so we know we are done with merging
            if (best_pair == []):
                break
        
            # now we have our best_pair, which tells us which pair we have to merge.
            # we just go down and merge each pair that matches the best_pair
            temp_word = []
            index = 0
            merge_pair_bytes = best_pair[0] + best_pair[1]
            while index < (len(merge_word)):
                if (index < len(merge_word) - 1) and (merge_word[index] == best_pair[0]) and (merge_word[index + 1] == best_pair[1]):
                    temp_word.append(merge_pair_bytes)
                    index += 2
                else:
                    temp_word.append(merge_word[index])
                    index += 1
            merge_word = temp_word
        return merge_word 

    def encode(self, text: str) -> list[int]:
        # convert the string to the base vocab plus special characters, not including merges (so utf-8 plus specials)
        # apply the merges to each word 
        # convert everything to the vocab indices

        # split on the length-sorted special tokens, this time keeping them in the list
        if (self.special_tokens is not None) and (len(self.special_tokens) != 0):
            pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            capture_pattern = f"({pattern})"
            captured_split_string = re.split(capture_pattern, text)
        else:
            captured_split_string = [text]

        gpt2_split_string = []
        # now split each string using the gpt2 regex pattern
        for line in captured_split_string:
            if line in self.special_tokens:
                gpt2_split_string.append(line)
            else:
                words = re.finditer(PAT, line)
                for word in words:
                    gpt2_split_string.append(word.group()) 
        encoded_text: list[int] = []
        # now encode to utf-8 and apply merges on each string in the list  except strings that are in our special_tokens
        for word in gpt2_split_string:
            if word not in self.special_tokens:
                encoded_substring = tuple(bytes([x]) for x in word.encode("utf-8"))
                encoded_bytes_word = self._apply_merges_word(encoded_substring)
                for byte in encoded_bytes_word:
                    encoded_text.append(self.reverse_vocab[byte])
            else:
                encoded_text.append(self.reverse_vocab[bytes(word.encode("utf-8"))])
        return encoded_text
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # lazily encode one chunk (e.g. one line of a file) at a time
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[x] for x in ids)
        return decoded_bytes.decode("utf-8", errors="replace")
