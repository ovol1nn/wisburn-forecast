from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import chunked_tslib_dataset as bridge


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
TSLIB_ROOT = Path(os.environ.get(
    "WISBURN_TSLIB_ROOT",
    WORKSPACE_ROOT / "offline" / "forecast" / "tslib",
))

# VSCode interactive defaults.
# 1) Run once with INTERACTIVE_RUN_TRAINING = False to generate/inspect metadata and command.
# 2) Set INTERACTIVE_RUN_TRAINING = True to launch original TSLib training.
INTERACTIVE_TRAINING_PROFILE = "model1_10s_trend_8gpu"  # model1_10s_trend_8gpu | model1_8gpu | model2_8gpu
VM_TRAINING_PROFILES = {
    "model1_10s_trend_8gpu": {
        "model_slug": "model1",
        "freq_seconds": 10,
        "seq_len": 120,
        "label_len": 60,
        "pred_len": 60,
        "epochs": 60,
        "patience": 8,
        "batch_size": 128,
        "learning_rate": 2e-5,
        "d_model": 192,
        "n_heads": 6,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 384,
        "patch_len": 8,
        "loss_trend_weight": 0.25,
        "loss_endpoint_weight": 0.5,
        "des": "xingrong3_model1_10s_trend_endpoint_v1",
    },
    "model1_8gpu": {
        "model_slug": "model1",
        "epochs": 8,
        "patience": 3,
        "batch_size": 128,
        "learning_rate": 2e-5,
        "d_model": 192,
        "n_heads": 6,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 384,
        "patch_len": 8,
        "des": "xingrong3_model1_pred600_8gpu_v2_constdrop",
    },
    "model2_8gpu": {
        "model_slug": "model2",
        "epochs": 8,
        "patience": 3,
        "batch_size": 256,
        "learning_rate": 3e-5,
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "patch_len": 16,
        "des": "xingrong3_model2_pred600_8gpu_v1",
    },
}
PROFILE = VM_TRAINING_PROFILES[INTERACTIVE_TRAINING_PROFILE]

INTERACTIVE_MODEL_SLUG = PROFILE["model_slug"]
INTERACTIVE_FREQ_SECONDS = PROFILE.get("freq_seconds", 1)
INTERACTIVE_SEQ_LEN = PROFILE.get("seq_len", 1200)
INTERACTIVE_LABEL_LEN = PROFILE.get("label_len", 600)
INTERACTIVE_PRED_LEN = PROFILE.get("pred_len", 600)
INTERACTIVE_FORCE_REFIT_SCALER = False
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
INTERACTIVE_DEVICES = "0,1,2,3,4,5,6,7"
INTERACTIVE_DES = PROFILE["des"]
INTERACTIVE_PRINT_EVERY = 100
INTERACTIVE_RESUME_TRAINING = False
INTERACTIVE_RESUME_CHECKPOINT = ""  # empty means checkpoints/<setting>/resume_state.pth


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_command(metadata: dict) -> list[str]:
    target_indices = ",".join(str(i) for i in metadata["target_indices"])
    root_path = (
        Path("..")
        / "outputs"
        / "training_ready"
        / metadata["dataset_id"]
    )
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
    if INTERACTIVE_USE_MULTI_GPU:
        cmd.extend(["--use_multi_gpu", "--devices", INTERACTIVE_DEVICES])
    return cmd


def write_command_file(path: Path, cmd: list[str]) -> None:
    path.write_text(" ".join(f'"{x}"' if " " in x else x for x in cmd) + "\n", encoding="utf-8")


def main() -> int:
    metadata = bridge.prepare_bridge(
        INTERACTIVE_MODEL_SLUG,
        freq_seconds=INTERACTIVE_FREQ_SECONDS,
        seq_len=INTERACTIVE_SEQ_LEN,
        label_len=INTERACTIVE_LABEL_LEN,
        pred_len=INTERACTIVE_PRED_LEN,
        force_refit_scaler=INTERACTIVE_FORCE_REFIT_SCALER,
    )
    cmd = build_command(metadata)
    ready_dir = bridge.training_ready_dir_for(
        metadata["model_slug"],
        int(metadata["freq_seconds"]),
        int(metadata["seq_len"]),
        int(metadata["pred_len"]),
    )
    command_path = ready_dir / "last_train_command.txt"
    write_command_file(command_path, cmd)

    print(f"training metadata: {rel_path(ready_dir / metadata['metadata_json'])}", flush=True)
    print(f"targets: {metadata['target_columns']}", flush=True)
    print(f"target_indices: {metadata['target_indices']}", flush=True)
    print(f"n_vars: {metadata['n_vars']}", flush=True)
    print(f"command file: {rel_path(command_path)}", flush=True)

    if not INTERACTIVE_RUN_TRAINING:
        print("INTERACTIVE_RUN_TRAINING = False; command prepared but not launched.", flush=True)
        return 0

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
    subprocess.run(cmd, cwd=str(TSLIB_ROOT), check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
