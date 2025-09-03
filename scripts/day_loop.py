# scripts/day_loop.py
# RQ2 driver: LoRA-only, Replay-only, Hybrid (LoRA + Replay 1:1)

import os, orjson, argparse, random, time, csv
from pathlib import Path
from glob import glob
import subprocess

# ------------- IO utils -------------
def load_jsonl(path, limit=None):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "rb") as f:
        for _, line in enumerate(f):
            rows.append(orjson.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

def dedupe_by_question(rows):
    """Remove duplicates by question-like key."""
    out, seen = [], set()
    for r in rows:
        q = r.get("question") or r.get("prompt") or r.get("input")
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(r)
    return out

def sample_mix(a, b, k_each):
    """Shuffle and take k from a and k from b."""
    a = list(a); b = list(b)
    random.shuffle(a); random.shuffle(b)
    return a[:k_each] + b[:k_each]

def run_cmd(args_list):
    print(">>>", " ".join(map(str, args_list)))
    subprocess.run(list(map(str, args_list)), check=True)

# ------------- Main loop -------------
def main(args):
    random.seed(42)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    # Discover day files
    day_train_files = sorted(glob(os.path.join(args.days_dir, "qa_train_day_*.jsonl")))
    day_eval_files  = sorted(glob(os.path.join(args.days_dir, "qa_eval_day_*.jsonl")))
    assert len(day_train_files) == len(day_eval_files) >= args.max_days, "Days missing"

    # Summary CSV
    metrics_csv = out_root / "summary.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["day","regimen","ffi","legacy","train_seconds","n_today","buffer_size"])
        w.writeheader()

    # Replay buffer
    buffer_path = out_root / "replay_buffer.jsonl"
    if args.reset and buffer_path.exists():
        buffer_path.unlink()

    # Helper: cap merged buffer by recent N (keep most recent)
    def cap_recent(merged):
        if args.replay_cap is not None and len(merged) > args.replay_cap:
            return merged[-args.replay_cap:]
        # If no cap provided, fall back to rate (for backward compat)
        if args.replay_cap is None and args.replay_rate is not None:
            k = max(1, int(round(args.replay_rate * len(merged))))
            # keep the most recent k
            return merged[-k:]
        return merged

    for d in range(1, args.max_days + 1):
        dd = f"{d:02d}"
        day_train = day_train_files[d-1]
        day_eval  = day_eval_files[d-1]
        day_tag   = f"day_{dd}"

        adapter_dir = Path(args.adapters_root) / args.regimen / day_tag
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # ---- Build / update replay buffer (CRITICAL: add today's before training) ----
        prev_buf = load_jsonl(buffer_path)
        today    = load_jsonl(day_train)
        # Merge, dedupe, then cap by recent
        merged = dedupe_by_question(prev_buf + today)
        merged = cap_recent(merged)
        save_jsonl(buffer_path, merged)
        buf_size = len(merged)
        n_today  = len(today)

        # ---- Choose training set by regimen ----
        if args.regimen == "lora_only":
            train_src   = day_train
            train_rows  = today

        elif args.regimen == "replay_only":
            # Train strictly on buffer which now includes today
            train_src   = str(buffer_path)
            train_rows  = merged

        elif args.regimen == "hybrid":
            # 1:1 today vs PREVIOUS-only (avoid double-counting today)
            prev_only = dedupe_by_question(prev_buf)
            if prev_only:
                k = min(len(today), len(prev_only))
                mix = sample_mix(today, prev_only, k_each=k)
            else:
                mix = today
            tmp_mix = out_root / f"hybrid_train_{day_tag}.jsonl"
            save_jsonl(tmp_mix, mix)
            train_src  = str(tmp_mix)
            train_rows = mix
        else:
            raise ValueError(f"Unknown regimen: {args.regimen}")

        # ---- Train LoRA adapter ----
        t0 = time.time()
        train_cmd = [
            "python", "scripts/train_lora.py",
            "--base_model", args.base_model,
            "--train_jsonl", train_src,
            "--out_dir", str(adapter_dir),
            "--epochs", str(args.epochs_per_day),
            "--bsz", str(args.bsz),
            "--grad_accum", str(args.grad_accum),
            "--lr", str(args.lr),
            "--max_len", str(args.max_len),
            "--device", args.device,
        ]
        if args.bf16: train_cmd.append("--bf16")
        if args.fp16: train_cmd.append("--fp16")
        if args.load_4bit: train_cmd.append("--load_4bit")
        if args.gradient_checkpointing: train_cmd.append("--gradient_checkpointing")

        run_cmd(train_cmd)
        train_seconds = time.time() - t0

        # ---- Evaluate: FFI (day-new) ----
        ffi_dir = out_root / "ffi" / day_tag
        ffi_dir.mkdir(parents=True, exist_ok=True)
        ffi_cmd = [
            "python", "scripts/eval_with_adapter.py",
            "--base_model", args.base_model,
            "--adapter_dir", str(adapter_dir),
            "--qa_file", day_eval,
            "--out_dir", str(ffi_dir),
            "--max_eval", str(args.max_eval),
            "--max_new_tokens", str(args.max_new_tokens),
            "--device", args.device,
        ]
        if args.bf16: ffi_cmd.append("--bf16")
        if args.fp16: ffi_cmd.append("--fp16")
        run_cmd(ffi_cmd)

        # ---- Evaluate: Legacy ----
        legacy_dir = out_root / "legacy" / day_tag
        legacy_dir.mkdir(parents=True, exist_ok=True)
        leg_cmd = [
            "python", "scripts/eval_with_adapter.py",
            "--base_model", args.base_model,
            "--adapter_dir", str(adapter_dir),
            "--qa_file", args.legacy_eval,
            "--out_dir", str(legacy_dir),
            "--max_eval", str(args.max_eval),
            "--max_new_tokens", str(args.max_new_tokens),
            "--device", args.device,
        ]
        if args.bf16: leg_cmd.append("--bf16")
        if args.fp16: leg_cmd.append("--fp16")
        run_cmd(leg_cmd)

        # ---- Read scores (with guard) ----
        def read_metric(p):
            try:
                with open(p, "rb") as f:
                    return orjson.loads(f.read()).get("exact_match", 0.0)
            except Exception:
                return 0.0

        ffi_score = read_metric(ffi_dir / "eval.json")
        leg_score = read_metric(legacy_dir / "eval.json")

        # ---- Log line + CSV ----
        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["day","regimen","ffi","legacy","train_seconds","n_today","buffer_size"])
            w.writerow({
                "day": d, "regimen": args.regimen,
                "ffi": round(ffi_score, 6), "legacy": round(leg_score, 6),
                "train_seconds": round(train_seconds, 1),
                "n_today": n_today, "buffer_size": buf_size
            })

        print(f"[{args.regimen} {day_tag}] FFI={ffi_score:.3f} Legacy={leg_score:.3f} "
              f"train_s={train_seconds:.1f} n_today={n_today} buf={buf_size}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--days_dir", default="data/days")
    ap.add_argument("--legacy_eval", default="data/legacy/qa_legacy_holdout.jsonl")
    ap.add_argument("--regimen", choices=["lora_only","replay_only","hybrid"], required=True)

    # Replay control:
    # Prefer replay_cap (absolute) for clarity; if None, we fall back to replay_rate.
    ap.add_argument("--replay_cap", type=int, default=2000,
                    help="Keep at most N most-recent items in the replay buffer (includes current day).")
    ap.add_argument("--replay_rate", type=float, default=None,
                    help="(Legacy) If replay_cap is None, keep last ceil(rate * total) items.")
    ap.add_argument("--reset", action="store_true",
                    help="Start with empty buffer (delete existing replay_buffer.jsonl).")

    # Training + eval
    ap.add_argument("--epochs_per_day", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_days", type=int, default=3)

    ap.add_argument("--adapters_root", default="models/adapters")
    ap.add_argument("--out_root", default="artifacts/rq2_run")

    # GPU / precision flags
    ap.add_argument("--device", choices=["auto","cuda","mps","cpu"], default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    args = ap.parse_args()
    main(args)
