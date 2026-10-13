import torch
import numpy as np

def data_loading(x: np.array, batch_size: int, context_length: int, device: torch.device | None = None) -> [torch.long, torch.long]:
    # return two tensors, the first being for training (strings length S, starting at the first token), and the second for gradients (ending at last token)
    # the entries in the outputs are integers and are token IDs
    # randomly decide what goes in each batch, and the sequential S tokens go into the sequence (well the 0:S-1, and 1:S in the second)
    # start by making a numpy array of length batch size, where each position tells us where to start in x to build our sequences of length S + 1
    sequence_start_pos = np.random.randint(0, len(x) - context_length, batch_size)
    # the last possible position at which we select a starting token is len(x) - batch_size - 1, that way we can have len(x) - 1, the final
    # index in x, be our last possible token in the sequence
    # build the output tensor:
    output_lists = []
    for start_pos in sequence_start_pos:
        output_lists.append(x[start_pos:start_pos + context_length + 1]) # include start_pos through context_length
    # turn the list of numpy arrays into a numpy array
    output_numpy_array = np.stack(output_lists).astype("int64")
    # the rows dimension are separate batches
    # we want to split within a batch, on the sequence dimension, which are the columns
    torch_output = torch.from_numpy(output_numpy_array).to(device)
    return [torch_output[..., 0:-1], torch_output[..., 1:]]
