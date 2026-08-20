from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from lite_dare.experiment_logger import (
    ConsoleCapture,
    TrainingRuntimeTracker,
    collect_environment_info,
    collect_parameter_statistics,
    utc_now_iso,
    write_json,
    write_run_manifest,
)
from lite_dare.extract_training_metrics import extract_run_metrics


DEFAULT_CONFIG_FILENAME = "train_exploration_transformer_node_discrete.yaml"

ORIGINAL_ENCODER_TARGET = (
    "diffusion_policy.model.encoder.exploration_node_encoder."
    "ExplorationNodeEncoder"
)
LITE_ENCODER_TARGET = (
    "lite_dare.lite_exploration_node_encoder."
    "LiteExplorationNodeEncoder"
)
EXPECTED_WORKSPACE_TARGET = (
    "diffusion_policy.workspace.train_diffusion_transformer_node_workspace."
    "TrainDiffusionTransformerNodeWorkspace"
)
EXPECTED_POLICY_TARGET = (
    "diffusion_policy.policy.diffusion_transformer_node_discrete_policy."
    "DiffusionTransformerNodeDiscretePolicy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain discrete-policy Lite-DARE L4 or L2 from scratch while "
            "keeping original DARE files and existing checkpoints unchanged."
        )
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        required=True,
        choices=(4, 2),
        help="Graph self-attention encoder depth.",
    )
    parser.add_argument(
        "--encoder-heads",
        type=int,
        default=4,
        help="Keep 4 for the controlled L6/L4/L2 comparison.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Discrete YAML path. Defaults to "
            "<DARE root>/train_exploration_transformer_node_discrete.yaml."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Hydra config-group directory. Defaults to "
            "<DARE root>/diffusion_policy/config."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <DARE root>/lite_dare/runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Single training seed shared by L4 and L2 for the controlled comparison.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 avoids the observed Docker /dev/shm DataLoader bus error.",
    )
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override applied before the retraining profile.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-metric-plots",
        action="store_true",
        help="Keep logs and CSV data but skip automatic PNG generation.",
    )
    return parser.parse_args()


def resolve_project_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    package_dir = Path(__file__).resolve().parent
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else package_dir.parent.resolve()
    )
    config_file = (
        args.config_file.resolve()
        if args.config_file is not None
        else (project_root / DEFAULT_CONFIG_FILENAME).resolve()
    )
    config_dir = (
        args.config_dir.resolve()
        if args.config_dir is not None
        else (project_root / "diffusion_policy" / "config").resolve()
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (package_dir / "runs").resolve()
    )

    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Discrete training YAML not found: {config_file}"
        )
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"Hydra config directory not found: {config_dir}"
        )

    policy_file = (
        project_root
        / "diffusion_policy"
        / "policy"
        / "diffusion_transformer_node_discrete_policy.py"
    )
    if not policy_file.is_file():
        raise FileNotFoundError(
            f"Discrete policy implementation not found: {policy_file}"
        )

    return project_root, config_file, config_dir, output_root


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def compose_discrete_config(
    config_file: Path,
    config_dir: Path,
    overrides: list[str],
) -> DictConfig:
    """
    Compose the root-level discrete YAML without modifying the repository.

    The root YAML references Hydra groups such as task/exploration_node.
    Therefore a temporary copy of diffusion_policy/config is created and the
    root-level YAML is inserted as the primary config.
    """
    register_resolvers()

    with tempfile.TemporaryDirectory(
        prefix="lite_dare_discrete_config_"
    ) as temp_directory:
        temporary_config_dir = Path(temp_directory) / "config"
        shutil.copytree(config_dir, temporary_config_dir)
        shutil.copy2(
            config_file,
            temporary_config_dir / config_file.name,
        )

        with initialize_config_dir(
            version_base=None,
            config_dir=str(temporary_config_dir),
            job_name="lite_dare_discrete_training",
        ):
            return compose(
                config_name=config_file.stem,
                overrides=overrides,
                return_hydra_config=False,
            )


def validate_discrete_config(cfg: DictConfig) -> None:
    workspace_target = str(OmegaConf.select(cfg, "_target_"))
    policy_target = str(OmegaConf.select(cfg, "policy._target_"))
    encoder_target = str(
        OmegaConf.select(cfg, "policy.obs_encoder._target_")
    )

    if workspace_target != EXPECTED_WORKSPACE_TARGET:
        raise ValueError(
            f"Unexpected workspace target: {workspace_target}"
        )
    if policy_target != EXPECTED_POLICY_TARGET:
        raise ValueError(
            "The selected YAML does not use the expected discrete node "
            f"policy: {policy_target}"
        )
    if encoder_target != ORIGINAL_ENCODER_TARGET:
        raise ValueError(
            f"Unexpected original observation encoder: {encoder_target}"
        )

    if int(cfg.policy.n_layer) != 8:
        raise ValueError("policy.n_layer must remain 8.")
    if int(cfg.policy.n_head) != 4:
        raise ValueError("policy.n_head must remain 4.")
    if int(cfg.policy.n_emb) != 256:
        raise ValueError("policy.n_emb must remain 256.")


def apply_lite_encoder(
    cfg: DictConfig,
    encoder_layers: int,
    encoder_heads: int,
) -> None:
    if encoder_layers not in {2, 4}:
        raise ValueError("This experiment trains only L4 and L2.")
    if encoder_heads != 4:
        raise ValueError(
            "encoder_heads must remain 4 for the controlled comparison."
        )

    embedding_dim = int(cfg.task.embedding_dim)
    if embedding_dim % encoder_heads != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} is not divisible by "
            f"encoder_heads={encoder_heads}."
        )

    with open_dict(cfg.policy.obs_encoder):
        cfg.policy.obs_encoder._target_ = LITE_ENCODER_TARGET
        cfg.policy.obs_encoder.encoder_n_layer = encoder_layers
        cfg.policy.obs_encoder.encoder_n_head = encoder_heads


def apply_retraining_profile(
    cfg: DictConfig,
    args: argparse.Namespace,
) -> None:
    """
    Force a clean 200-epoch retraining run.

    The discrete YAML contains resume=True, batch_size=128 and workers=4.
    These values are deliberately replaced with safe controlled-comparison
    defaults. Dedicated CLI arguments can change the defaults explicitly.
    """
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative.")
    if args.train_batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero.")
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("DataLoader worker counts cannot be negative.")

    with open_dict(cfg.training):
        cfg.training.resume = False
        cfg.training.seed = int(args.seed)
        cfg.training.num_epochs = int(args.epochs)
        cfg.training.lr_warmup_steps = int(args.warmup_steps)

    with open_dict(cfg.dataloader):
        cfg.dataloader.batch_size = int(args.train_batch_size)
        cfg.dataloader.num_workers = int(args.num_workers)
        cfg.dataloader.persistent_workers = False

    with open_dict(cfg.val_dataloader):
        cfg.val_dataloader.batch_size = int(args.val_batch_size)
        cfg.val_dataloader.num_workers = int(args.val_num_workers)
        cfg.val_dataloader.persistent_workers = False

    if "logging" in cfg:
        with open_dict(cfg.logging):
            cfg.logging.resume = False
            cfg.logging.id = None


def architecture_name(cfg: DictConfig) -> str:
    return (
        "DiscreteLiteDARE_"
        f"NodeEncSA_L{cfg.policy.obs_encoder.encoder_n_layer}"
        f"_H{cfg.policy.obs_encoder.encoder_n_head}"
        f"_D{cfg.task.embedding_dim}_DecL1_H4"
    )


def create_run_dir(
    output_root: Path,
    architecture: str,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / architecture / f"seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def update_logging_metadata(
    cfg: DictConfig,
    architecture: str,
) -> None:
    if "logging" not in cfg:
        return

    with open_dict(cfg.logging):
        cfg.logging.name = (
            f"{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}_"
            f"{architecture}"
        )
        cfg.logging.group = architecture
        cfg.logging.tags = [
            "Lite-DARE",
            "discrete-policy",
            "retrain-from-scratch",
            architecture,
        ]


def save_architecture_metadata(
    cfg: DictConfig,
    run_dir: Path,
    architecture: str,
    source_config_file: Path,
    args: argparse.Namespace,
) -> None:
    OmegaConf.save(
        config=cfg,
        f=run_dir / "lite_training_config.yaml",
    )
    shutil.copy2(
        source_config_file,
        run_dir / source_config_file.name,
    )

    metadata: dict[str, Any] = {
        "model_family": "DARE",
        "variant": "Discrete Lite-DARE",
        "architecture_name": architecture,
        "retrained_from_scratch": True,
        "checkpoint_resume_used": False,
        "source_config_file": str(source_config_file),
        "workspace_target": str(cfg._target_),
        "policy_target": str(cfg.policy._target_),
        "observation_encoder_target": str(
            cfg.policy.obs_encoder._target_
        ),
        "changed_component": "ExplorationNodeEncoder.encoder depth",
        "graph_encoder_layers": int(
            cfg.policy.obs_encoder.encoder_n_layer
        ),
        "graph_encoder_heads": int(
            cfg.policy.obs_encoder.encoder_n_head
        ),
        "embedding_dimension": int(cfg.task.embedding_dim),
        "graph_decoder_layers": 1,
        "graph_decoder_heads": 4,
        "diffusion_transformer_layers": int(cfg.policy.n_layer),
        "diffusion_transformer_heads": int(cfg.policy.n_head),
        "diffusion_transformer_embedding": int(cfg.policy.n_emb),
        "training_epochs": int(cfg.training.num_epochs),
        "training_seed": int(cfg.training.seed),
        "training_batch_size": int(cfg.dataloader.batch_size),
        "validation_batch_size": int(
            cfg.val_dataloader.batch_size
        ),
        "training_num_workers": int(cfg.dataloader.num_workers),
        "validation_num_workers": int(
            cfg.val_dataloader.num_workers
        ),
        "cli_arguments": vars(args),
    }
    write_json(run_dir / "architecture.json", metadata)


def print_summary(
    cfg: DictConfig,
    architecture: str,
    config_file: Path,
    run_dir: Path | None,
) -> None:
    print("=" * 78)
    print(f"Configuration:                 {config_file}")
    print(f"Architecture:                  {architecture}")
    print(f"Policy:                        {cfg.policy._target_}")
    print(
        "Graph encoder self-attention:  "
        f"L{cfg.policy.obs_encoder.encoder_n_layer} / "
        f"H{cfg.policy.obs_encoder.encoder_n_head} / "
        f"D{cfg.task.embedding_dim}"
    )
    print("Graph decoder:                 L1 / H4")
    print(
        "Diffusion Transformer:         "
        f"L{cfg.policy.n_layer} / H{cfg.policy.n_head} / "
        f"D{cfg.policy.n_emb}"
    )
    print(f"Training epochs:               {cfg.training.num_epochs}")
    print(f"Resume checkpoint:             {cfg.training.resume}")
    print(f"Training batch size:           {cfg.dataloader.batch_size}")
    print(
        "Validation batch size:         "
        f"{cfg.val_dataloader.batch_size}"
    )
    print(
        "Training DataLoader workers:   "
        f"{cfg.dataloader.num_workers}"
    )
    print(
        "Validation DataLoader workers: "
        f"{cfg.val_dataloader.num_workers}"
    )
    print("Training mode:                 from scratch")
    if run_dir is not None:
        print(f"New checkpoint run folder:     {run_dir}")
        print(
            "Persistent console log:        "
            f"{run_dir / 'training_console.log'}"
        )
    print("=" * 78)


def construct_workspace(
    cfg: DictConfig,
    run_dir: Path,
) -> Any:
    workspace_class = hydra.utils.get_class(str(cfg._target_))
    try:
        return workspace_class(cfg, output_dir=str(run_dir))
    except TypeError as exc:
        raise TypeError(
            "The workspace constructor did not accept "
            "(cfg, output_dir=...). Compare this call with train.py."
        ) from exc


def main() -> int:
    args = parse_args()
    (
        project_root,
        config_file,
        config_dir,
        output_root,
    ) = resolve_project_paths(args)
    os.chdir(project_root)

    cfg = compose_discrete_config(
        config_file=config_file,
        config_dir=config_dir,
        overrides=args.override,
    )
    validate_discrete_config(cfg)
    apply_lite_encoder(
        cfg,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
    )
    apply_retraining_profile(cfg, args)

    architecture = architecture_name(cfg)
    update_logging_metadata(cfg, architecture)

    if args.dry_run:
        print_summary(
            cfg,
            architecture,
            config_file,
            run_dir=None,
        )
        print("Dry run completed. No files were created.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(
        output_root,
        architecture,
        seed=int(cfg.training.seed),
    )
    save_architecture_metadata(
        cfg,
        run_dir,
        architecture,
        config_file,
        args,
    )

    environment = collect_environment_info(project_root)
    write_run_manifest(
        run_dir,
        command=[sys.executable, *sys.argv],
        environment=environment,
    )

    status_path = run_dir / "run_status.json"
    write_json(
        status_path,
        {
            "status": "initialising",
            "updated_at_utc": utc_now_iso(),
        },
    )

    console_log = run_dir / "training_console.log"
    runtime_tracker = TrainingRuntimeTracker()
    parameter_info: dict[str, Any] = {}
    runtime_info: dict[str, Any] = {}
    error_info: dict[str, Any] = {}
    exit_code = 0

    with ConsoleCapture(console_log):
        print_summary(
            cfg,
            architecture,
            config_file,
            run_dir,
        )
        try:
            workspace = construct_workspace(cfg, run_dir)
            parameter_info = collect_parameter_statistics(workspace)
            write_json(
                run_dir / "parameter_statistics.json",
                parameter_info,
            )

            write_json(
                status_path,
                {
                    "status": "running",
                    "updated_at_utc": utc_now_iso(),
                },
            )

            runtime_tracker.start()
            workspace.run()
            runtime_info = runtime_tracker.finish()

            write_json(
                status_path,
                {
                    "status": "completed",
                    "updated_at_utc": utc_now_iso(),
                },
            )
        except Exception as exc:
            runtime_info = runtime_tracker.finish()
            error_info = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at_utc": utc_now_iso(),
                    **error_info,
                },
            )
            print(error_info["traceback"], file=sys.stderr)
            exit_code = 1

        run_summary = {
            "status": "completed" if exit_code == 0 else "failed",
            "architecture_name": architecture,
            **parameter_info,
            **runtime_info,
            **error_info,
        }
        write_json(run_dir / "run_summary.json", run_summary)

        try:
            metric_summary = extract_run_metrics(
                run_dir,
                create_plots=not args.no_metric_plots,
            )
            print(
                "Comparison artifacts saved to: "
                f"{run_dir / 'comparison_artifacts'}"
            )
            print(
                "Detected metric records: "
                f"{metric_summary.get('record_count', 0)}"
            )
        except Exception:
            print(
                "Metric extraction failed, but raw logs and checkpoints "
                "have been preserved.",
                file=sys.stderr,
            )
            print(traceback.format_exc(), file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())