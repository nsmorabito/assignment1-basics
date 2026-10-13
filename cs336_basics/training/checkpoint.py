import torch

def save_checkpoint(model, optimizer, iteration, out):
    # dump the state from the model and optimizer into the file-like object out
    # use state_dict to save both the model and the optimizer to get their relevant states and use 
    # torch.save(obj, out) to dump obj into out (supported by pytorch)
    # torch.save supports dicts so we can just use that
    obj = {"model_weights": model.state_dict(), "optimizer_states": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)
    return

def load_checkpoint(src, model, optimizer) -> int:
    # load a checkpoint from src, 
    # recover the model and optimizer states from the checkpoint
    # return the iteration number that was saved to the checkpoint
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model_weights"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_states"])
    return checkpoint_dict["iteration"]
