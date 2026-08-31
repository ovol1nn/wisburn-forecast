from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get(
    "WISBURN_FORECAST_ARTIFACT_ROOT",
    SCRIPT_DIR.parents[2] / "offline" / "artifacts" / "forecast",
)) / "data_engineering"

# VSCode interactive defaults.
INTERACTIVE_MODEL_SLUG = "model1"  # model1 | model2
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_PRED_LEN = 60
INTERACTIVE_TRAIN_RATIO = 0.70
INTERACTIVE_VAL_RATIO = 0.15
INTERACTIVE_TEST_RATIO = 0.15
INTERACTIVE_PURGE_GAP_CHUNKS = 1
INTERACTIVE_VALIDATE_WINDOWS = True


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def as_int(value: str | int | float | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def validate_window_file(path: Path, seq_len: int, pred_len: int, freq_seconds: int) -> tuple[int, list[str]]:
    errors: list[str] = []
    rows_checked = 0
    first_rows: list[dict[str, str]] = []
    last_row: dict[str, str] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < 3:
                first_rows.append(row)
            last_row = row
    sample_rows = first_rows
    if last_row and last_row not in sample_rows:
        sample_rows.append(last_row)
    for row in sample_rows:
        rows_checked += 1
        start_pos = as_int(row["start_pos"])
        input_end_pos = as_int(row["input_end_pos"])
        target_start_pos = as_int(row["target_start_pos"])
        target_end_pos = as_int(row["target_end_pos"])
        input_start = parse_time(row["input_start_time"])
        input_end = parse_time(row["input_end_time"])
        target_start = parse_time(row["target_start_time"])
        target_end = parse_time(row["target_end_time"])
        ok = (
            input_end_pos - start_pos + 1 == seq_len
            and target_start_pos - input_end_pos == 1
            and target_end_pos - target_start_pos + 1 == pred_len
            and (input_end - input_start).total_seconds() == (seq_len - 1) * freq_seconds
            and (target_start - input_end).total_seconds() == freq_seconds
            and (target_end - target_start).total_seconds() == (pred_len - 1) * freq_seconds
        )
        if not ok:
            errors.append(f"bad window in {path.name}: {row}")
            break
    return rows_checked, errors


def split_rows(rows: list[dict[str, str]], train_ratio: float, val_ratio: float, purge_gap_chunks: int) -> list[dict[str, str]]:
    usable = [r for r in rows if as_int(r.get("windows")) > 0]
    usable.sort(key=lambda r: parse_time(r["core_start"]))
    total_windows = sum(as_int(r["windows"]) for r in usable)
    if total_windows <= 0:
        return []

    train_cut = total_windows * train_ratio
    val_cut = total_windows * (train_ratio + val_ratio)
    cumulative = 0
    out: list[dict[str, str]] = []
    for row in usable:
        next_cumulative = cumulative + as_int(row["windows"])
        if next_cumulative <= train_cut:
            split = "train"
        elif cumulative >= val_cut:
            split = "test"
        else:
            split = "val"
        item = dict(row)
        item["split"] = split
        out.append(item)
        cumulative = next_cumulative

    if purge_gap_chunks > 0 and len(out) > 2:
        for boundary in ("train", "val"):
            last_idx = max((i for i, r in enumerate(out) if r["split"] == boundary), default=-1)
            if last_idx >= 0:
                for j in range(last_idx - purge_gap_chunks + 1, last_idx + purge_gap_chunks + 1):
                    if 0 <= j < len(out):
                        out[j]["split"] = f"purge_after_{boundary}"
    return out


def main() -> int:
    dataset_id = f"{INTERACTIVE_MODEL_SLUG}_{INTERACTIVE_FREQ_SECONDS}s_seq{INTERACTIVE_SEQ_LEN}_pred{INTERACTIVE_PRED_LEN}"
    chunk_root = ARTIFACT_ROOT / "outputs" / "datasets" / "chunked" / dataset_id
    manifest_path = chunk_root / "manifest.csv"
    rows = read_csv(manifest_path)
    split_rows_out = split_rows(
        rows,
        INTERACTIVE_TRAIN_RATIO,
        INTERACTIVE_VAL_RATIO,
        INTERACTIVE_PURGE_GAP_CHUNKS,
    )
    if not split_rows_out:
        raise SystemExit(f"{dataset_id}: no windows available; build or fix chunked dataset first.")

    errors: list[str] = []
    checked = 0
    if INTERACTIVE_VALIDATE_WINDOWS:
        for row in split_rows_out:
            if not row["split"] in {"train", "val", "test"}:
                continue
            count, row_errors = validate_window_file(
                Path(row["window_path"]),
                INTERACTIVE_SEQ_LEN,
                INTERACTIVE_PRED_LEN,
                INTERACTIVE_FREQ_SECONDS,
            )
            checked += count
            errors.extend(row_errors)
            if errors:
                break
    if errors:
        raise SystemExit("\n".join(errors))

    split_dir = chunk_root / "splits"
    fieldnames = list(split_rows_out[0].keys())
    write_csv(split_dir / "split_manifest.csv", split_rows_out, fieldnames)
    summary = []
    for split in ("train", "val", "test"):
        subset = [r for r in split_rows_out if r["split"] == split]
        write_csv(split_dir / f"{split}_chunks.csv", subset, fieldnames)
        summary.append(
            {
                "split": split,
                "chunks": len(subset),
                "windows": sum(as_int(r["windows"]) for r in subset),
                "start": subset[0]["core_start"] if subset else "",
                "end": subset[-1]["core_end"] if subset else "",
            }
        )
    write_csv(split_dir / "split_summary.csv", summary)
    with (split_dir / "dataset_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": dataset_id,
                "seq_len": INTERACTIVE_SEQ_LEN,
                "pred_len": INTERACTIVE_PRED_LEN,
                "freq_seconds": INTERACTIVE_FREQ_SECONDS,
                "split_manifest": str(split_dir / "split_manifest.csv"),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"dataset split: {split_dir / 'split_manifest.csv'}", flush=True)
    print(f"validated sampled windows: {checked}", flush=True)
    for row in summary:
        print(f"{row['split']}: chunks={row['chunks']} windows={row['windows']} {row['start']} -> {row['end']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
