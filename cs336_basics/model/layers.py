import torch
from torch import nn
import math
from einops import einsum

# Linear is the model's basic learned matrix multiply, y = Mx, with no bias. It is the most reused part in the transformer:
# The attention projections Q, K, V, out
# the three weight matrices in SwiGlu
# the final LM head, which maps d_model to vocab_size
# 
# It owns one learnable weight W with shape (out_features, in_features). 
# Store W itself. 
# Initialize it with the truncated normal from 3.3.1 in the docs
# forward applies it to inputs with any number of leading batch dimensions, (..., in_features) -> (..., out_features)

# Linear is the simplest subclass of nn.Module that we make, and it just has one weights matrix
# I think later on we are going to define further subclasses of nn.Module that contain Linear objects as attributes, 
#       with more tensors, some that are parameters and some that aren't learnable
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        # sets up nn.Module's internals
        super().__init__()
        st_dev = math.sqrt(2 / (in_features + out_features))
        t = torch.empty(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(t, mean=0.0, std = st_dev, a = -3*st_dev, b = 3*st_dev)
        self.weight = nn.Parameter(t)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # we need to do x @ self.weights.transpose, which is equivalent to Wx in math notation
        # use einsum for this, and set d_out and d_in in the order they appear in the tensor
        # the input x can have more than one dimension, for example B, S, d_in. we want to sum over d_in and leave B, S unchanged
        #   this is how forward will handle leading dimensions separately
        # einsum takes care of this for us, we just represent the leading dimensions as ..., d_in, and the output as d_out. 
        # einsum then knows to sum over the leading dimensions and apply the linear transformation to d_in with d_out, d_in 
        # this gives us our desired linear transformation
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# subclass of nn.Module that handles embeddings.
# num_embeddings: the size of the vocab
# embedding_dim: the dimension of the embedding vectors, d_model
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        t = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(t, mean=0.0, std = 1.0, a = -3, b = 3)
        self.weight = nn.Parameter(t)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for each b,s in B,S dimensions, we have a single int which is our vocab token ID.
        # the goal of forward is to compute all the token embeddings in this (B, S) matrix.
        # doing this is as simple as looking up the embedding vector for each token ID.
        # I think the token ID corresponds to the index of the num_embeddings dimension side of the embedding matrix, returning the embedding_dim row.
        # in pytorch, when you index a tensor's first axis (num_embeddings dimension length) with a tensor of integers, you get those rows, 
        # and the index tensor's shape is kept. This is perfect for doing our lookup.
        # this line just looks at all the dimensions of x, and turns each entry into a row which is an embedding vector.
        # the final shape is then just x0 x x1 x ... d_model
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None =None, dtype: torch.dtype | None =None):
        # one learnable parameter, the gain
        super().__init__()
        t = torch.empty(d_model, device=device, dtype=dtype)
        torch.nn.init.ones_(t)
        self.weight = nn.Parameter(t)
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply RMSNorm to x, keeping the dimensions the same 
        # steps of RMSNorm:
        # save x.dtype and convert x to float32
        # compute the RMS over the last axis with keepdim=True, adding eps inside the sqrt
        # divide by the MRS and multiply by the gain
        # convert back to the saved dtype and return

        x_dtype = x.dtype
        x_float = x.to(torch.float32)

        # compute RMS on the d_model axis for every vector in the leading dimension, 
        # so we go from R^ B x S x d_model to R^ B x S (x 1, [float] for RMS, we keepdim).
        # this will be our RMS tensor as it must apply for every activation in B, S
        rms = torch.sqrt((x_float**2).mean(dim=-1, keepdim=True) + (self.eps))

        # now apply rms to every activiation using RMSNorm(a_i) = a_i/RMS(a) * g_i
        # where i is the index of the activation in a in d_model
        # RMS(a) is the entry that corresponds to the same location in the leading dimensions of rms and the activation tensor
        # g is our learned weight tensor which is of shape (d_model)
        # pytorch uses broadcasting, so the dimension of shape 1 gets repeated d_model times when we do our element-wise division,
        # and the leading dimensions line up, so everything matches.
        # then when it's time to multiply element-wise by g, pytorch uses broadcasting again to see the rightmost dimensions
        # of x and self.weight line up. Then pytorch broadcasts the elements of g to its missing dimensions to match the shape of x.
        # thus every single entry in B, S get a g vector. 
        # Then pytorch does element-wise multiplication on B,S, (activation vectors) and B,S, (g vector repeated)
        return ((x_float/rms)*self.weight).to(x_dtype)