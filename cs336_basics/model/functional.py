import torch
from einops import einsum
import math

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract off max element of the dimension from everything in that dimension
    # elementwise exp
    # reduction with keepdim=True
    # broadcasting
    x_max = x.amax(dim=dim, keepdim=True)
    new_x = x - x_max # I think this automatically broadcasts?
    x_exp = torch.exp(new_x)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    softmax = x_exp / x_exp_sum # this should also broadcast
    return softmax 

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # q is n x dk, K is m x dk, and v is m x dv, not learnable parameters (theyre a form of activations eg Wk_i * x)
    # we are transposing k on the last two dimensions seq_len and d_k so einsum is probably better
    # the mask seq_len x seq_len is applied in the softmax operation so we will separate this up 
    d_k = q.shape[-1]
    # all of these operations batch on anything preceeding seq_len_queries, so in MHST we put num_heads beforehand
    inner = einsum(q, k, "... seq_len_queries d_k, ... seq_len_keys d_k -> ... seq_len_queries seq_len_keys") / math.sqrt(d_k)
    # apply the boolean mask: True means do nothing to the attention index at ij, False means add -inf to it
    # had to ask claude how to apply this 
    masked_inner = torch.where(mask, inner, float("-inf"))
    # softmax the keys dimension, since that is what the value matrix uses for its operations
    # rows are identified by a query, so we don't want to softmax along that dimension, we want to softmax along the key, which is dim[-1]
    # we note that the values has the same starting dimension seq_len_keys as the last dimension in our softmaxxed inner, and that is the
    # one we want to collapse on. we keep the rows, of dimension length seq_len_queries, and the resulting values which is of course
    # dimension d_v (this is the same as d_model/h in our implementation, as attention is applied to the split up QKV tensors individually)
    softmax_inner = softmax(masked_inner, -1)
    return einsum(softmax_inner, v, "... seq_len_queries seq_len_keys, ... seq_len_keys d_v -> ... seq_len_queries d_v")
