## trainer.py
import os
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from training.utils.summary import summarize_training_data, print_module_summary
from training.utils.checkpoint import save_model_state, load_checkpoint

def get_log_steps(n_steps, max_logs=6):
    max_log_steps = min(n_steps // 100, max_logs)
    return [int(i) for i in torch.linspace(1, n_steps - 1, max_log_steps).tolist()][1:-1] + [n_steps - 1]

def get_dataloader(df, cfg, world_size, rank, is_train=True):
    from training.utils import import_utils
    dataset = import_utils.construct_class_by_name(
        df,
        class_name=cfg['dataset_class'],
        **cfg.get('dataset_class_kwargs', dict())
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        sampler=sampler,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    return loader, sampler

def train(rank, world_size, cfg, resume=None):
    import torch.distributed as dist
    import pandas as pd
    from training.utils import import_utils

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    log_file = os.path.join(cfg['run_dir'], "log.txt")
    train_df = pd.read_csv(cfg['train_data'])
    val_df = pd.read_csv(cfg['val_data']) if cfg['val_data'], False) else None

    # ==== Model/Loss/Acc objects (created from config) ====
    model = import_utils.construct_class_by_name(
        *cfg.get('model_class_args', tuple()),
        class_name=cfg['model_class'],
        **cfg.get('model_class_kwargs', dict())
    ).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = import_utils.construct_class_by_name(
        optimizer,
        *cfg.get('scheduler_class_args', tuple()),
        class_name=cfg['scheduler_class'],
        **cfg.get('scheduler_class_kwargs', dict())
    ) if cfg.get('scheduler_class', False) else None
    loss_obj = import_utils.construct_class_by_name(
        *cfg.get('loss_class_args', tuple()),
        class_name=cfg['loss_class'],
        **cfg.get('loss_class_kwargs', dict())
    ).to(device)
    accuracy_obj = import_utils.construct_class_by_name(
        *cfg.get('accuracy_class_args', tuple()),
        class_name=cfg['accuracy_class'],
        **cfg.get('accuracy_class_kwargs', dict())
    ).to(device)

    # ++++++++++++++ Print Summary +++++++++++++++++++++++++++++++++++
    if rank == 0:
        summarize_training_data(train_df, cfg, log_file)
        dummy_input = torch.randn(1, 1, *cfg['dataset_class_kwargs']['target_size'], device=device)  # adapt shape/channels as needed
        print_module_summary(model.module if hasattr(model, "module") else model, [dummy_input], log_file=log_file)

    start_epoch = 0
    # ==== Resume logic using new checkpoint style ====
    if os.path.exists(str(resume)):
        missing, unexpected, start_epoch, extra = load_checkpoint(
            cfg['run_dir'],
            model.module if hasattr(model, "module") else model,
            optimizer=optimizer,          # or None if not resuming opt/sched
            scheduler=scheduler,
            device=device,
            checkpoint_path=str(resume),
            strict=False,                # Allows loading even if shapes/heads changed
            rank= rank
        )

    train_loader, train_sampler = get_dataloader(train_df, cfg, world_size, rank, is_train=True)
    val_loader, val_sampler = None, None
    if val_df is not None:
        val_loader, val_sampler = get_dataloader(val_df, cfg, world_size, rank, is_train=False)

    total_epochs = cfg['epochs']

    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        acc_sums = {}
        n_batches = 0

        n_steps = len(train_loader)
        log_steps = set(get_log_steps(n_steps, max_logs=6))
        pbar = tqdm(enumerate(train_loader), total=n_steps, position=0, disable=rank != 0, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for step, (images, labels) in pbar:
            images = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_obj(preds, targets)
            metrics = accuracy_obj(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

            # Accumulate metrics for epoch-level averaging
            for k, v in metrics.items():
                acc_sums[k] = acc_sums.get(k, 0) + (0 if v is None or v != v else v)  # skip NaN

            if rank == 0:
                postfix = {'loss': f"{loss.item():.4f}"}
                postfix.update({k: f"{metrics[k]:.3f}" if metrics[k] is not None and metrics[k] == metrics[k] else "nan" for k in metrics})
                pbar.set_postfix(postfix)

            # Log to file at step, max 6 logs/epoch
            if rank == 0 and step in log_steps:
                log_str = (
                    f"[Train] Epoch {epoch} Step {step}/{n_steps} | "
                    + f"Loss: {loss.item():.4f} | "
                    + " | ".join(
                        f"{k}: {metrics[k]:.4f}" if metrics[k] is not None and metrics[k] == metrics[k] else f"{k}: nan"
                        for k in metrics
                    )
                )
                with open(log_file, "a") as f:
                    f.write(log_str + "\n")

        if rank == 0:
            avg_loss = epoch_loss / n_batches if n_batches else float('nan')
            avg_metrics = {k: (acc_sums[k] / n_batches if n_batches else float('nan')) for k in acc_sums}
            avg_metrics['loss'] = avg_loss
            log_str = (f"Epoch {epoch}: Loss {avg_loss:.4f}, " + ", ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items() if k != 'loss'))
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

        # Validation
        if val_loader is not None and epoch % cfg.get('val_freq', 5) == 0:
            model.eval()
            val_epoch_loss = 0
            val_acc_sums = {}
            val_n_batches = 0
            n_val_steps = len(val_loader)
            val_log_steps = set(get_log_steps(n_val_steps, max_logs=6))
            val_pbar = tqdm(enumerate(val_loader), total=n_val_steps, position=0, disable=rank != 0, desc=f"Val {epoch}", leave=False)
            with torch.no_grad():
                for val_step, (images, labels) in val_pbar:
                    images = images.to(device, non_blocking=True)
                    targets = labels.to(device, non_blocking=True)
                    preds = model(images)
                    loss = loss_obj(preds, targets)
                    metrics = accuracy_obj(preds, targets)
                    val_epoch_loss += loss.item()
                    val_n_batches += 1
                    for k, v in metrics.items():
                        val_acc_sums[k] = val_acc_sums.get(k, 0) + (0 if v is None or v != v else v)
                    if rank == 0:
                        postfix = {'loss': f"{loss.item():.4f}"}
                        postfix.update({k: f"{metrics[k]:.3f}" if metrics[k] is not None and metrics[k] == metrics[k] else "nan" for k in metrics})
                        val_pbar.set_postfix(postfix)
                    if rank == 0 and val_step in val_log_steps:
                        log_str = (
                            f"[Val] Epoch {epoch} Step {val_step}/{n_val_steps} | "
                            + f"Loss: {loss.item():.4f} | "
                            + " | ".join(
                                f"{k}: {metrics[k]:.4f}" if metrics[k] is not None and metrics[k] == metrics[k] else f"{k}: nan"
                                for k in metrics
                            )
                        )
                        with open(log_file, "a") as f:
                            f.write(log_str + "\n")
            if rank == 0:
                avg_val_loss = val_epoch_loss / val_n_batches if val_n_batches else float('nan')
                avg_val_metrics = {k: (val_acc_sums[k] / val_n_batches if val_n_batches else float('nan')) for k in val_acc_sums}
                avg_val_metrics['loss'] = avg_val_loss
                log_str = (f"[Val] Epoch {epoch}: Val Loss {avg_val_loss:.4f}, " + ", ".join(f"{k}: {v:.4f}" for k, v in avg_val_metrics.items() if k != 'loss'))
                print(log_str)
                with open(log_file, "a") as f:
                    f.write(log_str + "\n")

        # -- Save checkpoint --
        if epoch % cfg.get('save_freq', 5) == 0 and rank == 0:
            save_model_state(
                model.module if hasattr(model, "module") else model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(cfg['run_dir'], f"ckpt_epoch_{epoch:03d}.pth")
            )

    dist.destroy_process_group()
