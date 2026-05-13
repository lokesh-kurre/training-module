import os
import torch

def save_model_state(model, optimizer, scheduler, epoch, path, extra=None):
    """
    Save model, optimizer, scheduler, epoch (plus any extra) in a .pth file.
    Usage:
        save_model_state(model, optimizer, scheduler, epoch, "model.pth")
    """
    checkpoint = {
        'model_state': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
    }
    if extra is not None:
        checkpoint['extra'] = extra
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to {path}")

def find_latest_checkpoint(run_dir):
    """Find the latest available checkpoint in a run directory."""
    ckpts = [f for f in os.listdir(run_dir) if f.startswith('ckpt_epoch_') and f.endswith(('.pth', '.pkl'))]
    if not ckpts:
        return None
    ckpts.sort()
    return os.path.join(run_dir, ckpts[-1])

def load_checkpoint(run_dir, model, optimizer=None, scheduler=None, device="cpu", checkpoint_path=None, strict=False, rank= 0):
    """
    Load checkpoint and restore model/optimizer/scheduler state.
    Resumes from checkpoint_path if specified, else auto-finds latest in run_dir.

    Returns:
        missing_keys, unexpected_keys, epoch, extra (dict, often empty if none).
    """
    checkpoint_path = str(checkpoint_path)
    path = checkpoint_path if os.path.isfile(checkpoint_path) else find_latest_checkpoint(run_dir)
    if rank == 0 and not path or not os.path.isfile(path):
        print("No checkpoint found to resume from.")
        return [], [], 0, {}

    if rank == 0:
        print(f"{'Resuming' if os.path.isdir(checkpoint_path) else 'Loading Weights'} from checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint.get('model', checkpoint))  # supports both new/old
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if rank == 0 and missing_keys:
        print("Missing keys:", missing_keys)
    if rank == 0 and unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    if os.path.isdir(checkpoint_path):
        if optimizer is not None and 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        epoch = checkpoint.get('epoch', 0)
        extra = checkpoint.get('extra', {})
    else:
        epoch = 0
        extra = {}
    return missing_keys, unexpected_keys, epoch + 1, extra
