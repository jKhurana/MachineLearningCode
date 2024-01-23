import os
import shutil
from pathlib import Path
import torch

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    iteration,
    delete_before=None
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    iter_dir = os.path.join(checkpoint_dir, f'checkpoint-{iteration}')
    Path(iter_dir).mkdir(exist_ok=True)

    try:
        state_dict_model = model.module.state_dict()
    except AttributeError:
        state_dict_model = model.state_dict()

    torch.save(state_dict_model, os.path.join(iter_dir, 'pytorch_model.bin'))
    torch.save(optimizer.state_dict(), os.path.join(iter_dir, 'optimizer.pt'))

    if delete_before and delete_before > 0:
        for entry in os.listdir(checkpoint_dir):
            entry_iter = int(entry.split('-')[-1])
            if entry_iter < delete_before:
                shutil.rmtree(os.path.join(checkpoint_dir, entry))


def load_checkpoint(checkpoint_dir, model=None, optimizer=None, map_location=None, strict=True):
    kwargs = {}

    if map_location:
        kwargs['map_location'] = map_location

    if model:
        model_ckpt = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_ckpt, **kwargs), strict=strict)

    if optimizer:
        optimizer_ckpt = os.path.join(checkpoint_dir, 'optimizer.pt')
        optimizer.load_state_dict(torch.load(optimizer_ckpt, **kwargs))