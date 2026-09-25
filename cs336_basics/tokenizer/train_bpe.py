# the bpe trainer should first split up the big ahh string across all the special characters.
# then for each substring in our list[str] we pre-tokenize using gpt-2, giving us list[list[str]]
# then in each string at the very bottom, we split this into characters for utf-8 giving us list[list[list[utf-8 blocks]]], 
# then split those blocks into bytes, giving list[list[list[bytes]]] which is size |training set bytes| not including list library stuff
# now we have a whole bunch of lists of bytes, these individually are words. we will do our counts for byte pairs within words. 
# to do this, we build a large table, couting the frequency of identical pre-tokens for quicker computation:
# create word_count list[bytes]: count
# for each tuple of bytes word_bytes:
#   do this with a Counter (simpler)
#   append word_bytes: 1 to word_count if not in
#   else incrememnt word_count.get(word_bytes)
#
# now we have a map of tuples of bytes to count, and we can start our merges:
# create a starting vocab of the 256 individual bytes to ints, and special tokens to ints. 256 + number of special tokens in size. dict[int, bytes]
# create an empty merges list[tuple[bytes, bytes]]

# while len(vocab) < max_vocab:
#   create a potential_merges dict of pairs = tuple[bytes, bytes]: count
#   for each entry in word_count, word_bytes tuple[bytes]: count int:
#       for each pair b1, b2 of (pre-tokens, bytes)s in word_bytes, ie index 0 thru len(word_bytes)-2 and count:
#           do this with a Counter, not with traditional list stuff 
#           if b1,b2 not in potential_merges:
#               append (b1, b2): count
#           else potential_merges.get(b1, b2) += count 
#   if potential_merges is empty, stop = True, break;
#   merge():
#       find the max count's key m1, m2 with tiebreak logic 
#       add len(vocab): m1 + m2 to the vocab
#       in all the pre-tokens tuples, turn the pair b'm1', b'm2' into b'"m1" + "m2"', in left-to-right order
#       add tuple[m1, m2] to merges
# return vocab, merges


import os
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# step one: read in the stream of strings via the input file
def get_input_string(filename: str | os.PathLike) -> str:
    with open(filename, encoding="utf-8") as input_file:
        input_string = input_file.read()
    return input_string

# split the large input string on special tokens with regex
# claude pointed out to pass the prefix tests, we should sort the tokens that way we go in order of largest tokens to smallest
# and thus no token can be a prefix of the one after it.
def split_input_string(input_string: str, special_tokens: list[str]) -> list[str]:
    if len(special_tokens) != 0:
        pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        split_string = re.split(pattern, input_string)
        return split_string
    else:
        return [input_string]

# pre-tokenize each split input string with GPT2's regex pattern
# then count the ocurrences of each word and return a Counter[str]
def count_substrings(input_list: list[str]) -> Counter[str] :
    substring_counter = Counter()
    for split_string in input_list:
        # returns an iterator of Match objects
        split_substring = re.finditer(PAT, split_string)
        for word in split_substring:
            substring_counter[word.group()] += 1
    return substring_counter

# now we need a method to convert each counter of strings to a Counter of byte tuples
def string_to_bytes_counter(string_counter: Counter[str]) -> Counter[tuple[bytes, ...]]:
    byte_tuple_counter = Counter()
    for substring, count in string_counter.items():
        encoded_substring = tuple(bytes([x]) for x in substring.encode("utf-8"))
        byte_tuple_counter[encoded_substring] = count
    return byte_tuple_counter


# now we can write out our merge loop. We start with a counter of byte tuples and nothing else. 
# start by building the starting vocab, the 256 utf-8 bytes and custom tokens. the vocab is int: bytes.
def build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = dict[int, bytes]()
    # all the utf-8 bytes first
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    return vocab

# I believe there is a way to speed up count_pairs from it's naive implementation
# the docs say to cache all the pair counts
# then after a merge, update the pair counts incrimentally, since after a merge, the only pairs that are affected are ones that overlap with the merge
# but what do we do with something like merge(a, a) and there's an a, a, a? clearly this becomes aa, a. But we don't know how the counts change in this case.
# sounds like the best way to do this is update the bytes counter one word at a time, within merge.  we will do this with the helper function after count_pairs


def count_pairs_in_word_tuple(word_tuple: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    token_pair_counter = Counter[tuple[bytes, bytes]]()
    for index in range(len((word_tuple)) - 1):
        token_pair_counter[word_tuple[index], word_tuple[index + 1]] += 1
    return token_pair_counter 


# now we should have a helper function that counts all the pairs in the tuple[bytes, ...] counter and returns a Counter of all the possible byte pairs
def count_pairs(word_counter: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    token_pair_counter = Counter[tuple[bytes, bytes]]()
    for word_tuple in word_counter:
        for index in range(len((word_tuple)) - 1):
            token_pair_counter[word_tuple[index], word_tuple[index + 1]] += word_counter[word_tuple]
    return token_pair_counter

# takes in a pair counter, a pair counter to subtract, and a pair counter to add. updates the input pair counter in place.
# train_bpe can then use this function to update the pair counter, given we keep track of what was changed in the merge and the respective counts to add and remove
def update_count_pairs_during_merge(input_token_pair_counter: Counter[tuple[bytes, bytes]], remove: Counter[tuple[bytes, bytes]], add: Counter[tuple[bytes, bytes]]):
    for remove_pair in remove:
        input_token_pair_counter[remove_pair] -= remove[remove_pair]
    for add_pair in add:
        input_token_pair_counter[add_pair] += add[add_pair]


# then one that finds the pair of tokens that are most abundant, then sorted lexicographically
def find_max_token_pair(byte_pair_counter: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes] | None:
    if not byte_pair_counter:
        return None
    best_pair = max(byte_pair_counter, key = lambda pair: (byte_pair_counter[pair], pair))
    return best_pair

# now we have a vocab, counter of byte tuples, and a best pair of bytes to merge. Assume we also have a list of merges.
# time to do all the logic for the merge.
# we can assume the vocab and merges will be passed in externally and updated in place, since we aren't doing anything with the existing keys.
# the counter is being rewritten on each pass so it still must be returned.
# now that we are updating the pair counter in place as well, we need to know exactly what happened during the merge.
# In order to do this, merge has to return two Counter[tuple[bytes, bytes]], remove and add. 
# remove counts all the old pairs
# add counts all the new pairs
def merge(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], token_tuple_counter: Counter[tuple[bytes, ...]], max_token_pair: tuple[bytes, bytes]) -> [Counter[tuple[bytes, ...]], Counter[tuple[bytes, bytes]], Counter[tuple[bytes, bytes]]]:
    new_vocab_bytes = max_token_pair[0] + max_token_pair[1]
    vocab[len(vocab)] = new_vocab_bytes
    merges.append(max_token_pair)
    remove = Counter[tuple[bytes, bytes]]()
    add = Counter[tuple[bytes, bytes]]()
    new_token_tuple_counter = Counter[tuple[bytes, ...]]()
    # we can speed this up by skipping the new creation of new_token_list if our pair doesn't show up in the word (most times it won't!)
    # then we can just use the existing token_tuple add it to the counter
    for token_tuple in token_tuple_counter:
        
        pair_exists_in_tuple = False
        checker_index = 0
        while checker_index < len(token_tuple):
            if (checker_index < len(token_tuple) - 1) and (token_tuple[checker_index] == max_token_pair[0]) and (token_tuple[checker_index + 1] == max_token_pair[1]):
                pair_exists_in_tuple = True
                break;
            checker_index += 1
        word_count = token_tuple_counter[token_tuple]
        # if it does exist, merge it
        # update the remove and add tuples for the pair counter later in the train loop
        if pair_exists_in_tuple:
            new_token_list = []
            
            index = 0
            while (index < len(token_tuple)):
                if (index < len(token_tuple)- 1) and (token_tuple[index] == max_token_pair[0]) and (token_tuple[index + 1] == max_token_pair[1]):
                    new_token_list.append(new_vocab_bytes)
                    index += 2
                else:
                    new_token_list.append(token_tuple[index])
                    index += 1
            new_token_tuple = tuple[bytes, ...](new_token_list)
            new_token_tuple_counter[new_token_tuple] = word_count
            # make a token pair counter for the remove counter
            remove_counter = count_pairs_in_word_tuple(token_tuple)
            for pair in remove_counter:
                remove[pair] += remove_counter[pair] * word_count 
            add_counter = count_pairs_in_word_tuple(new_token_tuple)
            for pair in add_counter:
                add[pair] += add_counter[pair] * word_count
        else:
            new_token_tuple_counter[token_tuple] = word_count
    return [remove, add, new_token_tuple_counter]

# the train_bpe full loop - take a filepath, a max vocab size, and a list of special tokens and return the vocab dict and the list of merges
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens=None) -> [dict[int, bytes], list[tuple[bytes, bytes]]]:

    if special_tokens is None:
        special_tokens = []
    input_string = get_input_string(input_path)
    split_input_string_list  = split_input_string(input_string, special_tokens)
    input_substring_word_count = count_substrings(split_input_string_list)
    token_tuple_counter = string_to_bytes_counter(input_substring_word_count)

    vocab = build_initial_vocab(special_tokens)
    merges = []
    counted_pairs = count_pairs(token_tuple_counter)
    # now we have taken our input file and converted it to a counter of byte tuples. we have also defined our starting vocab.
    while len(vocab) < vocab_size:
        
        max_token_pair = find_max_token_pair(counted_pairs)
        if max_token_pair is None:
            break
        [remove, add, token_tuple_counter] = merge(vocab, merges, token_tuple_counter, max_token_pair)
        update_count_pairs_during_merge(counted_pairs, remove, add)
    return [vocab, merges]

train_bpe("tests/fixtures/corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])