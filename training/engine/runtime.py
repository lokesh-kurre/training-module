from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedRuntime:
    enabled: bool
    backend: str
    rank: int
    world_size: int
    local_rank: int
    device: str
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    sync_batchnorm: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.enabled and self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def build_distributed_runtime(cfg: dict[str, Any]) -> DistributedRuntime:
    """Build runtime state from config and standard torch.distributed env vars."""
    trainer_cfg = cfg.get("trainer", {})
    distributed_cfg = cfg.get("distributed", {})

    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    enabled = bool(trainer_cfg.get("distributed", False) or distributed_cfg.get("enabled", False) or world_size > 1)
    backend = str(trainer_cfg.get("distributed_backend", distributed_cfg.get("backend", "gloo")))
    device = str(trainer_cfg.get("device", distributed_cfg.get("device", "cpu")))

    return DistributedRuntime(
        enabled=enabled,
        backend=backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        find_unused_parameters=bool(trainer_cfg.get("find_unused_parameters", distributed_cfg.get("find_unused_parameters", False))),
        broadcast_buffers=bool(trainer_cfg.get("broadcast_buffers", distributed_cfg.get("broadcast_buffers", True))),
        sync_batchnorm=bool(trainer_cfg.get("sync_batchnorm", distributed_cfg.get("sync_batchnorm", False))),
    )


def setup_distributed(runtime: DistributedRuntime) -> None:
    """Initialize the process group only when distributed is actually requested."""
    if not runtime.is_distributed:
        return

    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=runtime.backend, init_method="env://")


def teardown_distributed(runtime: DistributedRuntime) -> None:
    """Tear down the process group only if it was created."""
    if not runtime.is_distributed:
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_runtime(model: Any, runtime: DistributedRuntime) -> Any:
    """Wrap the model in DDP when the runtime indicates multi-process training."""
    if not runtime.is_distributed:
        return model

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if runtime.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if runtime.device.startswith("cuda") and torch.cuda.is_available():
        local_device = int(runtime.local_rank)
        torch.cuda.set_device(local_device)
        return DistributedDataParallel(
            model,
            device_ids=[local_device],
            output_device=local_device,
            find_unused_parameters=runtime.find_unused_parameters,
            broadcast_buffers=runtime.broadcast_buffers,
        )

    return DistributedDataParallel(
        model,
        find_unused_parameters=runtime.find_unused_parameters,
        broadcast_buffers=runtime.broadcast_buffers,
    )


def reduce_scalar(value: float, runtime: DistributedRuntime) -> float:
    """Average a scalar across processes when DDP is enabled."""
    if not runtime.is_distributed:
        return value

    import torch
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(runtime.world_size)
    return float(tensor.item())
