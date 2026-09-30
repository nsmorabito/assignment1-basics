import torch

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        # no learned parameters, but we must calculate the Ri blocks via the sin and cos stuff from the docs
        # to avoid storing seq_len # of massive dxd sparse blocks, we can store d/2 * k 2x2 rotation matrices.
        # in fact better yet, we can just store what goes inside them, since we reuse each entry twice. 
        # so for every i,k in seq_len, 1..d/2, we store cos[i,k] and sin[i,k]
        # these things are stored as buffers.
        super().__init__()
        # calculating the table of theta_(i_k)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        per_pair_frequencies = torch.arange(1, d_k//2 + 1, device=device)
        per_pair_frequencies = 1 / (theta ** ((2 * per_pair_frequencies - 2)/d_k))
        # row i has a vector of length d_k/2 which corresponds to the theta_(i,k) value we want
        # theta_table[i, k] == theta_(i,k)
        theta_table = torch.outer(positions, per_pair_frequencies)
        cos_table = torch.cos(theta_table)
        sin_table = torch.sin(theta_table)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # process an input tensor of shape (..., sequence_length, d_k) and return a tensor of the same shape
        # x should be able to have an arbitrary number of batch dimensions
        # the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension
        # now we have a table of i,k cos and sin elements, and since the operations boil down mostly to element-wise operations
        # (2x2 matrix multiplication with rotation matrices is simple) we can do this efficiently 
        # part of how we do this is by splitting the token vectors by 0::2 and 1::2 (the last axis)

        # split the rightmost dimension into two parts
        x_2k = x[..., 0::2]
        x_2k_1 = x[..., 1::2]

        # index sin and cos by the token_positions
        # one row per position - ie reorder the rows by the token_positions vector
        # indexing looks at the first dimension of the tensor, unlike broadcasting which is right-aligned aritmetic between tensors
        # now we have token_positions ( = max_seq_length) rows of the sin and cos vectors of length d_k/2 
        # which is what we need for rope (we split input, with rows length d_k, in two)
        ordered_sin = self.sin_table[token_positions]
        ordered_cos = self.cos_table[token_positions]

        # x_2k and x_2k_1 are already ordered by token_positions, just as we have done with ordered_cos and ordered_sin
        # we presume the preceeding dimensions of x and sin/cos match
        # all the element-wise operations can be done safely now
        x_2k_rot = ordered_cos * x_2k - ordered_sin * x_2k_1
        x_2k_1_rot = ordered_sin * x_2k + ordered_cos * x_2k_1

        # interleave the rotated x vectors back together
        pairs = torch.stack([x_2k_rot, x_2k_1_rot], dim=-1)
        
        return pairs.flatten(-2)