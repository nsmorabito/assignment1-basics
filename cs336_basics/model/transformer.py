import torch
from .layers import Linear, Embedding, RMSNorm, SwiGLU, multihead_self_attention
from .functional import softmax

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, # required no matter what
    rms_epsilon: float | None = 1e-5, # RMSNorm
    theta: float | None = None, max_seq_len: int | None = None, # RoPE
    d_k: int | None = None, d_v: int | None = None, # very optional, if we want to customize QKV dim
    device=None, dtype=None):
        super().__init__()
        # build all our components:
        self.ln1 = RMSNorm(d_model, rms_epsilon, device, dtype)
        self.attn = multihead_self_attention(d_model, num_heads, d_k, d_v, theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(d_model, rms_epsilon, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype) 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        mhsa = self.attn(self.ln1(x), token_positions)
        mhsa_plus_x_residual = mhsa + x
        pwff = self.ffn(self.ln2(mhsa_plus_x_residual))
        return mhsa_plus_x_residual + pwff

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, # model parameters required
    num_heads: int, d_ff: int, # transformer block parameters required no matter what
    rms_epsilon: float | None = 1e-5, # RMSNorm
    theta: float | None = None, max_seq_len: int | None = None, # RoPE
    d_k: int | None = None, d_v: int | None = None, # very optional, if we want to customize QKV dim
    device=None, dtype=None):
        super().__init__()
        # build our components. 
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        transformer_module_list = []
        # we want RoPE on even if our max_seq_length isn't given (cuz it can be implied from the context_length)
        if max_seq_len is None:
            max_seq_len = context_length
        for i in range(0, num_layers):
            transformer_module_list.append(TransformerBlock(d_model, num_heads, d_ff, rms_epsilon, theta, max_seq_len, d_k, d_v, device, dtype))
        self.layers = torch.nn.ModuleList(transformer_module_list) 
        self.ln_final = RMSNorm(d_model, rms_epsilon, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype) # reverse the encoding, simple matrix multiply

        self.context_length = context_length
        self.num_layers = num_layers

    def forward(self, in_indices: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # in_indices tells us which vocab token IDs to run our predictions on. shape (B, S) where S <= context_length
        # embedding self.weight[x] brings an ID into a d_model vector
        x = self.token_embeddings(in_indices) # shape (B, S, d_model)
        # if token_positions is None, we still want RoPE to run, so just make it arange(context_length) by default
        if token_positions is None:
            token_positions = torch.arange(0, in_indices.shape[-1]).to(x.device)
        # iterate over the torch.nn.ModuleList layers, which is just a list of transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
        # shape (B, S, d_model)
        x = self.ln_final(x) # shape (B, S, d_model) (normalized for the output encoding weights)
        return self.lm_head(x) # we want unnormalized predictions for each token, shape (B, S, vocab_size)
  