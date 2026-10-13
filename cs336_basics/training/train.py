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

# helper function to get the validation loss during a step
@torch.no_grad() # don't touch the gradients and fuck up the training loop
def get_validation_loss(model, val_arr, batch_size: int, context_length: int, num_batches: int | None = 25, device: torch.device | None = None) -> float:
    # run over several batches to avoid noisy loss data
    running_loss_total = 0
    for _ in range(num_batches):
        train_tensor, test_tensor = data_loading(val_arr, batch_size, context_length, device)
        logits = model(train_tensor)
        running_loss_total += cross_entropy_loss(logits, test_tensor).item()
    return running_loss_total / num_batches

# The training loop should save the model to a file via checkpointing, along with checkpoints along the way
# we assume we start with a training set that is already tokenized via our BPE tokenizer
# I guess we take in a lot of hyperparameters also, namely the ones that go in the params of the optimizer
def train(data_dir: str, train_file: str, val_file: str, context_length: int, 
d_model: int, num_layers: int, num_heads: int, d_ff: int, 
theta: float, lr: float, batch_size: int, num_steps: int, 
min_lr: float, max_lr: float, T_warmup: int, T_final: int,
betas: tuple[float, float] | None = [0.9, 0.999], weight_decay: float | None = 0.1, eps: float | None = 1e-8,
gradient_clipping_norm: float | None = 1.0, 
save_dir: str | None = None,
device: torch.device | None = None):
    # CO AUTHORED BY CLAUDE
    # snapshot every argument passed to train() so we can save it as the run's config (must be the first line)
    config = dict(locals())
    # pick a default device, mac so we default to mps otherwise cpu
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if save_dir is None:
        save_dir = data_dir
    # read vocab_size from data_dir
    # note that the text we are to train on has already been converted to tokens, by assumption
    with open(f"{data_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    # CO AUTHORED BY CLAUDE
    # record the derived values too, load_model needs vocab_size to rebuild the embedding/lm_head shapes
    config["vocab_size"] = vocab_size
    config["device"] = device

    # TransformerLM now should have all it needs, initalize it.
    transformerLM = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta = theta, device=device)
    # construct the input activation which will be used each time during forwards
    # first we need to load in the input list of token integers (x). 
    # numpy load with mmap_mode="r" handles not putting all of this to memory, since we only need some parts at a time
    x_train_arr = np.load(f"{data_dir}/{train_file}", mmap_mode = "r")
    x_val_arr = np.load(f"{data_dir}/{val_file}", mmap_mode = "r")
    
    # COMMENT OUT THIS LINE FOR ANYTHING THAT ISN'T THE OVERFITTING TEST
    # train_tensor, test_tensor = data_loading(x_train_arr, batch_size, context_length, device)

    # initialize the optimizer
    adam_optim = AdamW(transformerLM.parameters(), lr, betas, weight_decay, eps)
    # loop over steps with lr adjustment and gradient clipping
    for step in range(num_steps):

        # create the data loader for the train array - random at each step (UNCOMMENT THIS LINE FOR ANYTHING THAT ISN'T THE OVERFITTING TEST)
        train_tensor, test_tensor = data_loading(x_train_arr, batch_size, context_length, device)

        logits = transformerLM(train_tensor)
        # zero out the old gradients
        transformerLM.zero_grad()

        # compute the loss
        loss = cross_entropy_loss(logits, test_tensor)

        # pytorch keeps track of what has been done to the logits, since these were created by the transformerLM.
        # so loss.backwards runs on pytorch tensors, which are nn.parameters, so they store the history of the functions that have been applied
        # computes the gradients in transformerLM, gradient clipping then just has to look at transformerLM parameters to do its thang
        loss.backward() 

        # adjust the learning rate based on the global step count (not how many times an individual param was updated)
        adj_lr = learning_rate_schedule(step, max_lr, min_lr, T_warmup, T_final)

        
        if step % 20 == 0:
            # print the current validation loss and the current adjusted learning rate, as well as the current training loss
            val_loss = get_validation_loss(transformerLM, x_val_arr, batch_size, context_length, num_batches=10, device=device)
            print(f"Iteration {step} of {num_steps}: ----------------------")
            print(f"Validation Loss: {val_loss}, Training Loss: {loss.item()}, Learning Rate: {round(adj_lr, 5)}")
        
        for group in adam_optim.param_groups: # (really usually just one group per model, and each group has a lr, betas, epislon)
            group["lr"] = adj_lr
        gradient_clipping(transformerLM.parameters(), gradient_clipping_norm, eps)
        
        # remember, our optimizer knows our model, and we have adjusted the grads properly, so it has all it needs to update the weights
        adam_optim.step()

    # CO AUTHORED BY CLAUDE
    # save the checkpoint and the config together in their own run directory, named by the PST date and time
    run_dir = os.path.join(save_dir, datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_checkpoint(transformerLM, adam_optim, num_steps, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved checkpoint and config to {run_dir}")
    return run_dir

# CO AUTHORED BY CLAUDE
# rebuild a trained TransformerLM from a run directory written by train(), ready to run forward and get logits
# authored by claude cuz I'm too lazy to write this
def load_model(run_dir: str, device: torch.device | None = None) -> tuple[TransformerLM, dict]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    # construct the model with the exact same shapes it was trained with, then fill in the weights
    model = TransformerLM(config["vocab_size"], config["context_length"], config["num_layers"], config["d_model"],
                          config["num_heads"], config["d_ff"], theta=config["theta"], device=device)
    # load_checkpoint also restores an optimizer, so give it a throwaway one built over this model's parameters
    load_checkpoint(os.path.join(run_dir, "model.pt"), model, AdamW(model.parameters()))
    model.eval()
    return model, config
        
def main():
    d_model = 256
    train("data/tokenizers/tinystories_10k", "TinyStoriesV2-GPT4-train.npy", "TinyStoriesV2-GPT4-valid.npy", 
    context_length=128, d_model=d_model, num_layers=6, num_heads=8, d_ff=704, theta=10000.0, batch_size=32, num_steps=400, 
    lr=3e-3, max_lr=3e-3, min_lr=3e-4, T_warmup=20, T_final=400, weight_decay=0.1, betas=[0.9, 0.99], gradient_clipping_norm = 1.0)

if __name__ == "__main__":
    main()