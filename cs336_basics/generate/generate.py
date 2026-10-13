import torch
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import pickle
import math
# CO AUTHORED BY CLAUDE
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.tokenizer import Tokenizer

from cs336_basics.model.layers import Linear, Embedding, RMSNorm, SwiGLU, multihead_self_attention

from cs336_basics.model.functional import silu, softmax, scaled_dot_product_attention, cross_entropy_loss

from cs336_basics.model.rope import RotaryPositionEmbedding

from cs336_basics.model.transformer import TransformerBlock, TransformerLM

from cs336_basics.training.optim import AdamW, learning_rate_schedule, gradient_clipping

from cs336_basics.training.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.training.data import data_loading
# CO AUTHORED BY CLAUDE
import argparse
from cs336_basics.training.train import load_model

# let's just assume we have a string that does not include special tokens
def input_string_to_int_tensor(tokenizer, prompt: str, device: torch.device | None = None) -> torch.tensor:
    return torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device = device)


def top_p_sampling(probs, top_p) -> torch.tensor:
    # scale the logits basedd on the top_p rule 
    # renormalize the top_p number of tokens with the highest probability, 0 out the rest
    # do this by sorting the logits, getting the indices with the highest probability
    sorted_probabilities, vocab_indices = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probabilities, dim=-1)
    drop = (cum - sorted_probabilities) >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(drop, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probabilities, num_samples = 1)
    return torch.gather(vocab_indices, -1, choice)
    
@torch.no_grad
def generate(input_int_tensor, tokenizer, model, max_predictions: int, 
temperature: float | None = 1.0, top_p: int | None = 1) -> torch.tensor:
    # loop over model.forward until we either hit a max length of tokens or a special token
    special_tokens = tokenizer.special_tokens
    context_length = model.context_length
    eof_token_id = tokenizer.reverse_vocab["<|endoftext|>".encode("utf-8")]
    last_token_id = None
    prediction_count = 0
    curr_input_tensor = input_int_tensor
    while ((last_token_id is None) or (last_token_id != eof_token_id)) and (prediction_count < max_predictions):
        logits = model(curr_input_tensor[:, -context_length:])[:, -1, :]
        probs = softmax(logits / temperature, dim=-1)
        next_id = top_p_sampling(probs, top_p)
        curr_input_tensor = torch.cat([curr_input_tensor, next_id], dim=1)
        prediction_count += 1
        last_token_id = next_id.item()
    return curr_input_tensor # the full tensor including the starting input 

def generate_string(tokenizer, prompt: str, model, max_predictions: int, 
temperature: float | None = 1.0, top_p: int | None = 1) -> str:
    device = next(model.parameters()).device
    int_tensor = input_string_to_int_tensor(tokenizer, prompt, device)
    output_tensor = generate(int_tensor, tokenizer, model, max_predictions, temperature, top_p)
    return tokenizer.decode(output_tensor[0].tolist())

# CO AUTHORED BY CLAUDE
# a run directory (written by train()) holds model.pt + config.json, and config["data_dir"] points at the
# tokenizer folder (vocab.pkl, merges.pkl, special_tokens.pkl), so the run directory alone is enough to load both
def load_model_and_tokenizer(run_dir: str, tokenizer_dir: str | None = None):
    model, config = load_model(run_dir)
    if tokenizer_dir is None:
        tokenizer_dir = config["data_dir"]
    tokenizer = Tokenizer.from_files(
        os.path.join(tokenizer_dir, "vocab.pkl"),
        os.path.join(tokenizer_dir, "merges.pkl"),
        special_tokens_filepath=os.path.join(tokenizer_dir, "special_tokens.pkl"),
    )
    return model, tokenizer

# CO AUTHORED BY CLAUDE
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)  # e.g. data/tokenizers/tinystories_10k/2026-10-05_02-16-50
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tokenizer_dir", type=str, default=None)  # defaults to the data_dir saved in the run's config
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.run_dir, args.tokenizer_dir)
    output = generate_string(tokenizer, args.prompt, model, args.max_tokens, args.temperature, args.top_p)
    print(output)
    return output

if __name__ == "__main__":
    main()