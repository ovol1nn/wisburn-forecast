from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import chunked_tslib_dataset as bridge


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
TSLIB_ROOT_OVERRIDE = os.environ.get("WISBURN_TSLIB_ROOT")
TSLIB_ROOT_CANDIDATES = [
    SCRIPT_DIR / "Time-Series-Library",
    WORKSPACE_ROOT / "offline" / "forecast" / "tslib",
]
if TSLIB_ROOT_OVERRIDE:
    TSLIB_ROOT = Path(TSLIB_ROOT_OVERRIDE)
else:
    TSLIB_ROOT = next(
        (path for path in TSLIB_ROOT_CANDIDATES if (path / "run.py").exists()),
        TSLIB_ROOT_CANDIDATES[0],
    )

# VSCode interactive defaults.
# 1) Run once with INTERACTIVE_RUN_TRAINING = False to generate/inspect metadata and command.
# 2) Set INTERACTIVE_RUN_TRAINING = True to launch original TSLib training.
INTERACTIVE_TRAINING_PROFILE = "model1_10s"  # model1_10s | model2_10s
VM_TRAINING_PROFILES = {
    "model1_10s": {
        "model_slug": "model1",
        "freq_seconds": 10,
        "seq_len": 120,
        "label_len": 60,
        "pred_len": 60,
        "epochs": 60,
        "patience": 8,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "d_model": 192,
        "n_heads": 6,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 384,
        "patch_len": 8,
        "loss_trend_weight": 0.25,
        "loss_endpoint_weight": 0.5,
        "no_future_covariates": True,
        "des": "xingrong3_model1_10s_history_only_v1",
    },
    "model2_10s": {
        "model_slug": "model2",
        "freq_seconds": 10,
        "seq_len": 120,
        "label_len": 60,
        "pred_len": 60,
        "epochs": 60,
        "patience": 8,
        "batch_size": 64,
        "learning_rate": 3e-5,
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "patch_len": 16,
        "no_future_covariates": False,
        "des": "xingrong3_model2_10s_v1",
    },
}
PROFILE = VM_TRAINING_PROFILES[INTERACTIVE_TRAINING_PROFILE]

INTERACTIVE_MODEL_SLUG = PROFILE["model_slug"]
INTERACTIVE_FREQ_SECONDS = PROFILE.get("freq_seconds", 10)
INTERACTIVE_SEQ_LEN = PROFILE.get("seq_len", 120)
INTERACTIVE_LABEL_LEN = PROFILE.get("label_len", 60)
INTERACTIVE_PRED_LEN = PROFILE.get("pred_len", 60)
INTERACTIVE_FORCE_REFIT_SCALER = True
INTERACTIVE_RUN_TRAINING = False

# Original model training arguments.
INTERACTIVE_TSLIB_MODEL = "Crossformer"  # Crossformer | iTransformer | TimeXer | ...
INTERACTIVE_FEATURES = "M"
INTERACTIVE_EPOCHS = PROFILE["epochs"]
INTERACTIVE_PATIENCE = PROFILE["patience"]
INTERACTIVE_BATCH_SIZE = PROFILE["batch_size"]
INTERACTIVE_LEARNING_RATE = PROFILE["learning_rate"]
INTERACTIVE_NUM_WORKERS = 0
INTERACTIVE_USE_AMP = True
INTERACTIVE_FAIL_ON_NONFINITE = True
INTERACTIVE_CHUNK_MAX_TRAIN_WINDOWS = 0  # 0 means full train split
INTERACTIVE_CHUNK_MAX_EVAL_WINDOWS = 0  # 0 means full val/test split
INTERACTIVE_D_MODEL = PROFILE["d_model"]
INTERACTIVE_N_HEADS = PROFILE["n_heads"]
INTERACTIVE_E_LAYERS = PROFILE["e_layers"]
INTERACTIVE_D_LAYERS = PROFILE["d_layers"]
INTERACTIVE_D_FF = PROFILE["d_ff"]
INTERACTIVE_FACTOR = 3
INTERACTIVE_PATCH_LEN = PROFILE["patch_len"]
INTERACTIVE_LOSS_TREND_WEIGHT = PROFILE.get("loss_trend_weight", 0.0)
INTERACTIVE_LOSS_ENDPOINT_WEIGHT = PROFILE.get("loss_endpoint_weight", 0.0)
INTERACTIVE_USE_GPU = True
INTERACTIVE_GPU = 0
INTERACTIVE_USE_MULTI_GPU = True
INTERACTIVE_DEVICES = "0,1"
INTERACTIVE_DES = PROFILE["des"]
INTERACTIVE_PRINT_EVERY = 100
INTERACTIVE_RESUME_TRAINING = False
INTERACTIVE_RESUME_CHECKPOINT = ""  # empty means checkpoints/<setting>/resume_state.pth


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_command(metadata: dict, ready_dir: Path) -> list[str]:
    target_indices = ",".join(str(i) for i in metadata["target_indices"])
    root_path = ready_dir.resolve()
    model_id = f"{metadata['model_slug']}_{INTERACTIVE_TSLIB_MODEL}_s{metadata['seq_len']}_p{metadata['pred_len']}"
    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        INTERACTIVE_TSLIB_MODEL,
        "--data",
        "custom_chunked",
        "--root_path",
        str(root_path),
        "--data_path",
        "metadata.json",
        "--features",
        INTERACTIVE_FEATURES,
        "--target",
        metadata["target_columns"][0],
        "--freq",
        "s",
        "--seq_len",
        str(metadata["seq_len"]),
        "--label_len",
        str(metadata["label_len"]),
        "--pred_len",
        str(metadata["pred_len"]),
        "--enc_in",
        str(metadata["n_vars"]),
        "--dec_in",
        str(metadata["n_vars"]),
        "--c_out",
        str(metadata["n_vars"]),
        "--d_model",
        str(INTERACTIVE_D_MODEL),
        "--n_heads",
        str(INTERACTIVE_N_HEADS),
        "--e_layers",
        str(INTERACTIVE_E_LAYERS),
        "--d_layers",
        str(INTERACTIVE_D_LAYERS),
        "--d_ff",
        str(INTERACTIVE_D_FF),
        "--factor",
        str(INTERACTIVE_FACTOR),
        "--patch_len",
        str(INTERACTIVE_PATCH_LEN),
        "--train_epochs",
        str(INTERACTIVE_EPOCHS),
        "--patience",
        str(INTERACTIVE_PATIENCE),
        "--batch_size",
        str(INTERACTIVE_BATCH_SIZE),
        "--learning_rate",
        str(INTERACTIVE_LEARNING_RATE),
        "--num_workers",
        str(INTERACTIVE_NUM_WORKERS),
        "--chunk_max_train_windows",
        str(INTERACTIVE_CHUNK_MAX_TRAIN_WINDOWS),
        "--chunk_max_eval_windows",
        str(INTERACTIVE_CHUNK_MAX_EVAL_WINDOWS),
        "--loss_target_indices",
        target_indices,
        "--loss_trend_weight",
        str(INTERACTIVE_LOSS_TREND_WEIGHT),
        "--loss_endpoint_weight",
        str(INTERACTIVE_LOSS_ENDPOINT_WEIGHT),
        "--des",
        INTERACTIVE_DES,
        "--itr",
        "1",
        "--gpu",
        str(INTERACTIVE_GPU),
        "--print_every",
        str(INTERACTIVE_PRINT_EVERY),
    ]
    if INTERACTIVE_RESUME_TRAINING:
        cmd.append("--resume")
        if INTERACTIVE_RESUME_CHECKPOINT:
            cmd.extend(["--resume_checkpoint", INTERACTIVE_RESUME_CHECKPOINT])
    if INTERACTIVE_USE_AMP:
        cmd.append("--use_amp")
    if INTERACTIVE_FAIL_ON_NONFINITE:
        cmd.append("--fail_on_nonfinite")
    if PROFILE.get("no_future_covariates", False):
        cmd.append("--no_future_covariates")
    if INTERACTIVE_USE_MULTI_GPU:
        cmd.extend(["--use_multi_gpu", "--devices", INTERACTIVE_DEVICES])
    return cmd


def write_command_file(path: Path, cmd: list[str]) -> None:
    path.write_text(" ".join(f'"{x}"' if " " in x else x for x in cmd) + "\n", encoding="utf-8")


def run_logged_training(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    started_at = datetime.now().astimezone()
    started_clock = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"started_at: {started_at.isoformat(timespec='seconds')}\n")
        log_file.write(f"workspace: {WORKSPACE_ROOT}\n")
        log_file.write(f"dataset_root: {bridge.chunk_root_for(INTERACTIVE_MODEL_SLUG, INTERACTIVE_FREQ_SECONDS, INTERACTIVE_SEQ_LEN, INTERACTIVE_PRED_LEN)}\n")
        log_file.write(f"tslib_root: {TSLIB_ROOT}\n")
        log_file.write(f"command: {' '.join(cmd)}\n")
        log_file.write("\n--- training output ---\n")

        print(f"training log: {log_path}", flush=True)
        process = subprocess.Popen(
            cmd,
            cwd=str(TSLIB_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)

        return_code = process.wait()
        elapsed_seconds = time.monotonic() - started_clock
        finished_at = datetime.now().astimezone()
        log_file.write("\n--- training summary ---\n")
        log_file.write(f"finished_at: {finished_at.isoformat(timespec='seconds')}\n")
        log_file.write(f"elapsed_seconds: {elapsed_seconds:.1f}\n")
        log_file.write(f"exit_code: {return_code}\n")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def validate_training_runtime() -> None:
    run_script = TSLIB_ROOT / "run.py"
    if not run_script.exists():
        raise SystemExit(
            f"Time-Series-Library run.py not found: {run_script}. "
            "Set WISBURN_TSLIB_ROOT to the directory containing run.py."
        )
    if not INTERACTIVE_USE_GPU:
        return
    if bridge.torch is None or not bridge.torch.cuda.is_available():
        raise SystemExit("GPU training requested, but PyTorch cannot access CUDA.")
    if INTERACTIVE_USE_MULTI_GPU:
        try:
            device_ids = [int(item.strip()) for item in INTERACTIVE_DEVICES.split(",") if item.strip()]
        except ValueError as exc:
            raise SystemExit(f"Invalid INTERACTIVE_DEVICES: {INTERACTIVE_DEVICES}") from exc
        if len(device_ids) < 2 or len(set(device_ids)) != len(device_ids):
            raise SystemExit("Multi-GPU mode needs at least two distinct IDs, e.g. '0,1'.")
        visible_count = bridge.torch.cuda.device_count()
        if max(device_ids) >= visible_count:
            raise SystemExit(
                f"Requested GPUs {INTERACTIVE_DEVICES}, but PyTorch sees only {visible_count}. "
                "Check nvidia-smi and CUDA_VISIBLE_DEVICES."
            )
        if INTERACTIVE_GPU != device_ids[0]:
            raise SystemExit("INTERACTIVE_GPU must match the first ID in INTERACTIVE_DEVICES.")


def main() -> int:
    metadata = bridge.prepare_bridge(
        INTERACTIVE_MODEL_SLUG,
        freq_seconds=INTERACTIVE_FREQ_SECONDS,
        seq_len=INTERACTIVE_SEQ_LEN,
        label_len=INTERACTIVE_LABEL_LEN,
        pred_len=INTERACTIVE_PRED_LEN,
        force_refit_scaler=INTERACTIVE_FORCE_REFIT_SCALER,
    )
    ready_dir = bridge.training_ready_dir_for(
        metadata["model_slug"],
        int(metadata["freq_seconds"]),
        int(metadata["seq_len"]),
        int(metadata["pred_len"]),
    )
    cmd = build_command(metadata, ready_dir)
    command_path = ready_dir / "last_train_command.txt"
    write_command_file(command_path, cmd)

    print(f"training metadata: {rel_path(ready_dir / metadata['metadata_json'])}", flush=True)
    print(f"targets: {metadata['target_columns']}", flush=True)
    print(f"target_indices: {metadata['target_indices']}", flush=True)
    print(f"n_vars: {metadata['n_vars']}", flush=True)
    print(f"command file: {rel_path(command_path)}", flush=True)
    dataset_root = bridge.chunk_root_for(
        INTERACTIVE_MODEL_SLUG,
        INTERACTIVE_FREQ_SECONDS,
        INTERACTIVE_SEQ_LEN,
        INTERACTIVE_PRED_LEN,
    )
    print(f"dataset root: {dataset_root}", flush=True)
    print(f"TSLib root: {TSLIB_ROOT}", flush=True)

    if not INTERACTIVE_RUN_TRAINING:
        bridge.smoke_test(metadata)
        print("INTERACTIVE_RUN_TRAINING = False; command prepared but not launched.", flush=True)
        return 0

    validate_training_runtime()

    env = os.environ.copy()
    if not INTERACTIVE_USE_GPU:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif INTERACTIVE_USE_MULTI_GPU:
        env["CUDA_VISIBLE_DEVICES"] = INTERACTIVE_DEVICES
    elif "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(INTERACTIVE_GPU)

    mpl_config_dir = WORKSPACE_ROOT / ".tmp" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_config_dir)


    print("launching Time-Series-Library training...", flush=True)
    print(" ".join(cmd), flush=True)
    run_started_at = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = ready_dir / "training_logs" / f"{INTERACTIVE_MODEL_SLUG}_{run_started_at}.log"
    run_logged_training(cmd, log_path, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
