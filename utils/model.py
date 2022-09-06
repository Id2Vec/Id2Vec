import os
import torch


def save_model(model, dir: str) -> None:
    """
    Save model to a specific dir.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(dir, "model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)