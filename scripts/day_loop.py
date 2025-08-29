# scripts/day_loop.py
# RQ2 driver: LoRA-only, Replay-only, Hybrid (LoRA + Replay 1:1)
import os
import sys
import orjson
import argparse
import random
import time
import subprocess
from pathlib import Path
from glob import glob

def load_jsonl(path, limit=None):
    rows = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            rows.append(orjson.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

def save_jsonl(path, rows):
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r)); f.write(b"\n")

def reservoir_sample(stream_rows, k):
    """Standard reservoir sampling to keep k items."""
    if k <= 0:
        return []
    sample = []
    for i, item in enumerate(stream_rows):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample

def run_cmd(cmd):
    """Print and run a subprocess command, coercing all args to str."""
    cmd = [str(c) for c in cmd]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main(args):
    random.seed(42)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    day_train_files = sorted(glob(os.path.join(args.days_dir, "qa_train_day_*.jsonl")))
    day_eval_files  = sorted(glob(os.path.join(args.days_dir, "qa_eval_day_*.jsonl")))
    assert len(day_train_files) == len(day_eval_files), "Mismatched train/eval day files"
    assert len(day_train_files) >= args.max_days, f"Need at least {args.max_days} days, found {len(day_train_files)}"

    metrics_csv = Path(args.out_root) / "summary.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    with open(metrics_csv, "w") as f:
        f.write("day,regimen,FFI,LegacyAcc,train_seconds,n_train,buffer_size\n")

    # Persistent replay buffer across days
    buffer_path = Path(args.out_root) / "replay_buffer.jsonl"
    if buffer_path.exists():
        buffer_path.unlink()

    for d in range(1, args.max_days + 1):
        day_train = day_train_files[d - 1]
        day_eval  = day_eval_files[d - 1]
        day_tag   = f"day_{d:02d}"
        adapter_dir = Path(args.adapters_root) / args.regimen / day_tag
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Build train set depending on regimen
        if args.regimen == "lora_only":
            train_src = day_train
            train_rows = load_jsonl(train_src)

            # Update buffer (not used for training in this regimen, but useful for later analysis)
            if args.replay_rate > 0:
                prev = load_jsonl(buffer_path) if buffer_path.exists() else []
                new_buf = prev + train_rows
                cap = max(args.min_replay_cap, int(args.replay_rate * len(new_buf)))
                sampled = reservoir_sample(new_buf, cap)
                save_jsonl(buffer_path, sampled)

        elif args.regimen == "replay_only":
            # Update buffer with today's rows
            today = load_jsonl(day_train)
            prev = load_jsonl(buffer_path) if buffer_path.exists() else []
            new_buf = prev + today
            cap = max(args.min_replay_cap, int(args.replay_rate * len(new_buf)))
            sampled = reservoir_sample(new_buf, cap)
            save_jsonl(buffer_path, sampled)

            # Train on buffer only
            train_src = buffer_path
            train_rows = load_jsonl(train_src)

        elif args.regimen == "hybrid":
            # Update buffer
            today = load_jsonl(day_train)
            prev = load_jsonl(buffer_path) if buffer_path.exists() else []
            new_buf = prev + today
            cap = max(args.min_replay_cap, int(args.replay_rate * len(new_buf)))
            sampled = reservoir_sample(new_buf, cap)
            save_jsonl(buffer_path, sampled)

            # 1:1 hybrid mix (clip to the smaller of the two)
            k = min(len(today), len(sampled))
            mix = today[:k] + sampled[:k]
            tmp_mix = Path(args.out_root) / f"hybrid_train_{day_tag}.jsonl"
            save_jsonl(tmp_mix, mix)
            train_src = str(tmp_mix)
            train_rows = mix
        else:
            raise ValueError("Unknown regimen: " + str(args.regimen))

        # Train adapter
        t0 = time.time()
        run_cmd([
            sys.executable, "scripts/train_lora.py",
            "--base_model", args.base_model,
            "--train_jsonl", train_src,
            "--out_dir", adapter_dir,
            "--epochs", args.epochs_per_day,
            "--bsz", args.bsz,
            "--grad_accum", args.grad_accum,
            "--lr", args.lr,
            "--max_len", args.max_len,
        ])
        train_seconds = time.time() - t0

        # Evaluate FFI (day-new eval)
        ffi_dir = Path(args.out_root) / "ffi" / day_tag
        ffi_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([
            sys.executable, "scripts/eval_with_adapter.py",
            "--base_model", args.base_model,
            "--adapter_dir", adapter_dir,
            "--qa_file", day_eval,
            "--out_dir", ffi_dir,
            "--max_eval", args.max_eval,
            "--max_new_tokens", "32",
        ])
        with open(ffi_dir / "eval.json", "rb") as f:
            ffi = orjson.loads(f.read())["exact_match"]

        # Evaluate Legacy (fixed set)
        legacy_dir = Path(args.out_root) / "legacy" / day_tag
        legacy_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([
            sys.executable, "scripts/eval_with_adapter.py",
            "--base_model", args.base_model,
            "--adapter_dir", adapter_dir,
            "--qa_file", args.legacy_eval,
            "--out_dir", legacy_dir,
            "--max_eval", args.max_eval,
            "--max_new_tokens", "32",
        ])
        with open(legacy_dir / "eval.json", "rb") as f:
            legacy = orjson.loads(f.read())["exact_match"]

        buf_size = 0
        if buffer_path.exists():
            with open(buffer_path, "rb") as bf:
                for _ in bf:
                    buf_size += 1

        with open(metrics_csv, "a") as f:
            f.write(f"{d},{args.regimen},{ffi:.4f},{legacy:.4f},{train_seconds:.1f},{len(train_rows)},{buf_size}\n")

        print(f"[{args.regimen} {day_tag}] FFI={ffi:.3f} Legacy={legacy:.3f} "
              f"train_s={train_seconds:.1f} n={len(train_rows)} buf={buf_size}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--days_dir", default="data/days")
    ap.add_argument("--legacy_eval", default="data/legacy/qa_legacy_holdout.jsonl")
    ap.add_argument("--regimen", choices=["lora_only", "replay_only", "hybrid"], required=True)

    # Replay settings
    ap.add_argument("--replay_rate", type=float, default=0.03, help="Fraction of seen examples to keep in buffer")
    ap.add_argument("--min_replay_cap", type=int, default=128, help="Minimum buffer size cap (for early days)")

    # Training/eval knobs
    ap.add_argument("--epochs_per_day", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--max_days", type=int, default=30)

    # Paths
    ap.add_argument("--adapters_root", default="models/adapters")
    ap.add_argument("--out_root", default="artifacts/rq2_run")

    args = ap.parse_args()
    main(args)
