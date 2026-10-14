# CO AUTHORED BY CLAUDE
# Model size specifications from Table 1 of the assignment 2 handout (mostly GPT-2 configs).
# Every benchmark shares these defaults unless a command-line flag overrides them:
# vocab size 10,000, batch size 4, context length 512.
from dataclasses import dataclass

VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
ROPE_THETA = 10_000.0


@dataclass(frozen=True)
class ModelSize:
    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZES: dict[str, ModelSize] = {
    size.name: size
    for size in [
        ModelSize("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        ModelSize("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        ModelSize("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        ModelSize("xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
        ModelSize("10B", d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
    ]
}


# parameter count of our TransformerLM for a given size, computed on paper (no allocation needed)
def count_parameters(size: ModelSize, vocab_size: int = VOCAB_SIZE) -> int:
    # per block: Q, K, V, output projections (4 d^2) + SwiGLU w1, w2, w3 (3 d d_ff) + ln1, ln2 gains (2 d)
    per_block = 4 * size.d_model**2 + 3 * size.d_model * size.d_ff + 2 * size.d_model
    # token embeddings + lm_head (separate matrices, not tied) + ln_final gain
    return size.num_layers * per_block + 2 * vocab_size * size.d_model + size.d_model


# rough fp32 training footprint before activations: weights + grads + AdamW m + AdamW v = 4 copies x 4 bytes
def training_state_gb(size: ModelSize, vocab_size: int = VOCAB_SIZE) -> float:
    return count_parameters(size, vocab_size) * 16 / 1e9
