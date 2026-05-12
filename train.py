from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from training.config import load_layered_config
from training.engine import Trainer
from training.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training module skeleton CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Execution mode (train, val, test)",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name from cfg/tasks")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name from cfg/datasets")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name from cfg/experiments/<task>",
    )
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        default=[],
        help="Config file path(s), applied after standard layers (primary runtime input)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values using dot notation, e.g. trainer.epochs=50",
    )
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training mode")
    parser.add_argument("--distributed-backend", type=str, default=None, help="Distributed backend (gloo, nccl)")
    parser.add_argument("--find-unused-parameters", action="store_true", help="Enable DDP find_unused_parameters")
    parser.add_argument("--sync-batchnorm", action="store_true", help="Convert batch norm layers to SyncBatchNorm")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run directory name")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and persist config without training")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    load_dotenv(override=False)

    args = parse_args()
    project_root = Path(__file__).resolve().parent

    cfg = load_layered_config(
        root_dir=project_root,
        task=args.task,
        dataset=args.dataset,
        experiment=args.experiment,
        extra_config_paths=[Path(p) for p in args.config],
        cli_overrides=args.overrides,
    )

    # Keep project metadata aligned with explicit CLI inputs.
    project_cfg = cfg.setdefault("project", {})
    if args.task:
        project_cfg["task"] = args.task
    if args.dataset:
        project_cfg["dataset"] = args.dataset
    if args.experiment:
        project_cfg["experiment"] = args.experiment

    if args.distributed:
        cfg.setdefault("distributed", {})["enabled"] = True
    if args.distributed_backend:
        cfg.setdefault("distributed", {})["backend"] = args.distributed_backend
    if args.find_unused_parameters:
        cfg.setdefault("distributed", {})["find_unused_parameters"] = True
    if args.sync_batchnorm:
        cfg.setdefault("distributed", {})["sync_batchnorm"] = True

    trainer = Trainer(cfg=cfg, project_root=project_root)
    if args.mode == "train":
        trainer.fit(run_name=args.run_name, dry_run=args.dry_run)
    elif args.mode == "val":
        trainer.validate(run_name=args.run_name, dry_run=args.dry_run)
    elif args.mode == "test":
        trainer.test(run_name=args.run_name, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    logger = get_logger("training.entrypoint.train")
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in train entrypoint")
        raise
