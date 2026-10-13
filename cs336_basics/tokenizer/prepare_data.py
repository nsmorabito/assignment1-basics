from email.policy import default
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.tokenizer import Tokenizer
import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

default_special_tokens = ["<|endoftext|>"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_path", type=str, required=True)
    parser.add_argument("--input_valid_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_tokens_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    # args.input_train_path, args.input_valid_path, args.vocab_size, etc are now available
    # handle special_tokens_path later, right now just use the default list ["<|endoftext|>"]
    if args.special_tokens_path is None:    
        special_tokens = default_special_tokens
    test_file_name = Path(args.input_train_path).stem
    valid_file_name = Path(args.input_valid_path).stem
    # create the encoding once, and save it for future use. use train_bpe to make our encoding
    vocab, merges = train_bpe(args.input_train_path, args.vocab_size, special_tokens, show_progress=True)
    token_izer = Tokenizer(vocab, merges, special_tokens)

    # (create the directory if it doesn't exist, it probably doesn't)
    os.makedirs(args.output_dir, exist_ok=True)
    token_izer.save_tokenizer_pieces(args.output_dir)

    # now we create both encoded files - train and valid, both on the encoding from the train input file.
    # we need to make it an iterable becuase the file is most likely huge
    with open(args.input_train_path, encoding="utf-8") as f:
        train_int_array = np.fromiter(token_izer.encode_iterable(tqdm(f, desc=f"encoding {test_file_name}", unit=" lines")), dtype = np.uint16)
    # save the encoded int string to a .npy file in our out_dir for the training loop
    np.save(f"{args.output_dir}/{test_file_name}", train_int_array)
    with open(args.input_valid_path, encoding="utf-8") as f:
        valid_int_array = np.fromiter(token_izer.encode_iterable(tqdm(f, desc=f"encoding {valid_file_name}", unit=" lines")), dtype = np.uint16)
    # save the encoded int string to a .npy file in our out_dir for the training loop
    np.save(f"{args.output_dir}/{valid_file_name}", valid_int_array)


if __name__ == "__main__": main()