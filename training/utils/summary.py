from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from training.utils.input_spec import resolve_input_spec
from training.utils.logger import get_logger


LOGGER = get_logger("training.utils.summary")


def summarize_training_data(train_data: Any, cfg: dict[str, Any], log_file: str | None = None) -> str:
    """Create a compact training data summary for logs."""
    lines: list[str] = []
    lines.append("=" * 42)
    lines.append("               DATA SUMMARY               ")
    lines.append("=" * 42)

    dataloader_cfg = cfg.get("dataloader", {})
    dataset_cfg = cfg.get("dataset", {})
    trainer_cfg = cfg.get("trainer", {})

    # Prefer explicit record count from generator datasets.
    n_samples: Any = getattr(train_data, "num_records", None)
    if n_samples is None:
        try:
            n_samples = len(train_data)
        except Exception:
            n_samples = "unknown"

    lines.append(f"Samples            | {n_samples}")
    lines.append(f"Dataloader backend | {dataloader_cfg.get('backend', 'torch')}")
    lines.append(f"Dataset backend    | {dataset_cfg.get('backend', 'torch')}")
    lines.append(f"Batch size         | {dataloader_cfg.get('batch_size', 32)}")
    lines.append(f"Num workers        | {dataloader_cfg.get('num_workers', 0)}")

    steps_cfg = trainer_cfg.get("steps_per_epoch", trainer_cfg.get("max_steps_per_epoch"))
    if steps_cfg is not None:
        est_steps = int(steps_cfg)
    else:
        try:
            est_steps = len(train_data)
        except Exception:
            est_steps = "unknown"
    lines.append(f"Estimated steps    | {est_steps}")

    try:
        input_size, input_layout = resolve_input_spec(cfg)
    except Exception:
        input_size, input_layout = ("?", "?"), "?"

    lines.append(f"Input size         | {input_size}")
    lines.append(f"Input layout       | {input_layout}")

    model_cfg = cfg.get("model", {})
    n_classes = model_cfg.get("num_classes", cfg.get("num_classes", "unknown"))
    lines.append(f"Num classes        | {n_classes}")

    columns = getattr(train_data, "columns", None)
    if columns is not None:
        if "type" in columns:
            try:
                type_counts = dict(train_data["type"].value_counts())
            except Exception:
                type_counts = "unknown"
            lines.append(f"Type counts        | {type_counts}")
        if "quality" in columns:
            try:
                quality = train_data["quality"]
                quality_range = f"{quality.min()} - {quality.max()}"
            except Exception:
                quality_range = "unknown"
            lines.append(f"Quality range      | {quality_range}")

    output = "\n".join(lines)
    LOGGER.info("\n%s", output)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(output + "\n")
    return output


def print_module_summary(
    module: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    log_file: str | None = None,
    max_nesting: int = 3,
    skip_redundant: bool = True,
) -> Any:
    """Print a minimal table-style module summary and return model outputs."""
    assert isinstance(module, torch.nn.Module)
    entries: list[dict[str, Any]] = []
    nesting = [0]

    def pre_hook(_mod: torch.nn.Module, _inputs: Any) -> None:
        nesting[0] += 1

    def post_hook(mod: torch.nn.Module, _inputs: Any, outputs: Any) -> None:
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outs = [tensor for tensor in outs if isinstance(tensor, torch.Tensor)]
            entries.append({"mod": mod, "outputs": outs})

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks.extend(mod.register_forward_hook(post_hook) for mod in module.modules())

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            outputs = module(*inputs)
    finally:
        module.train(was_training)

    for hook in hooks:
        hook.remove()

    tensors_seen: set[int] = set()
    for entry in entries:
        entry["unique_params"] = [t for t in entry["mod"].parameters(recurse=False) if id(t) not in tensors_seen]
        entry["unique_buffers"] = [t for t in entry["mod"].buffers(recurse=False) if id(t) not in tensors_seen]
        entry["unique_outputs"] = [t for t in entry["outputs"] if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in entry["unique_params"] + entry["unique_buffers"] + entry["unique_outputs"]}

    if skip_redundant:
        entries = [e for e in entries if e["unique_params"] or e["unique_buffers"] or e["unique_outputs"]]

    rows: list[list[str]] = [["Module", "Parameters", "Buffers", "Output shape", "Dtype"], ["---"] * 5]
    param_total = 0
    buffer_total = 0
    submodule_names = {submod: name for name, submod in module.named_modules()}
    for entry in entries:
        name = "<top-level>" if entry["mod"] is module else submodule_names[entry["mod"]]
        param_size = sum(t.numel() for t in entry["unique_params"])
        buffer_size = sum(t.numel() for t in entry["unique_buffers"])
        output_shapes = [str(list(t.shape)) for t in entry["outputs"]] or ["-"]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in entry["outputs"]] or ["-"]

        rows.append([name, str(param_size) if param_size else "-", str(buffer_size) if buffer_size else "-", output_shapes[0], output_dtypes[0]])
        for idx in range(1, len(output_shapes)):
            rows.append([f"{name}:{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]])

        param_total += param_size
        buffer_total += buffer_size

    rows.append(["---"] * 5)
    rows.append(["Total", str(param_total), str(buffer_total), "-", "-"])

    widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    output = "\n" + "\n".join(
        "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in rows
    ) + "\n"
    LOGGER.info("%s", output)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(output + "\n")

    return outputs
