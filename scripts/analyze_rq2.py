# scripts/analyze_rq2.py
import argparse
from pathlib import Path
import re
import sys
import json
from typing import List, Tuple, Optional, Dict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


REQ_COLS = ["day", "FFI", "LegacyAcc"]


def pretty_label(dir_name: str) -> str:
    name = dir_name.lower()

    if "lora_only" in name:
        base = "LoRA-only"
    elif "replay_only" in name:
        base = "Replay-only"
    elif "hybrid" in name:
        base = "Hybrid"
    else:
        base = dir_name  # fallback

    # smoke vs full
    suffix = " (smoke)" if "smoke" in name else " (full)"
    return f"{base}{suffix}"


def parse_day_from_dir(p: Path) -> Optional[int]:
    m = re.search(r"day_(\d+)", p.name)
    if m:
        return int(m.group(1))
    return None


def read_eval_json(eval_path: Path) -> Optional[float]:
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # We only need exact_match
        if isinstance(data, dict) and "exact_match" in data:
            return float(data["exact_match"])
    except Exception:
        pass
    return None


def rebuild_summary_from_evals(run_dir: Path) -> Optional[pd.DataFrame]:
    """Try to rebuild a summary dataframe from ffi/legacy eval.json files."""
    ffi_root = run_dir / "ffi"
    legacy_root = run_dir / "legacy"
    if not ffi_root.exists() or not legacy_root.exists():
        return None

    rows: List[Dict] = []
    # enumerate day folders from ffi and legacy; keep intersection
    ffi_days = {parse_day_from_dir(p): p for p in ffi_root.glob("day_*") if p.is_dir()}
    legacy_days = {parse_day_from_dir(p): p for p in legacy_root.glob("day_*") if p.is_dir()}
    common_days = sorted(d for d in ffi_days.keys() if d in legacy_days and d is not None)

    for day in common_days:
        ffi_eval = ffi_days[day] / "eval.json"
        legacy_eval = legacy_days[day] / "eval.json"
        ffi = read_eval_json(ffi_eval)
        legacy = read_eval_json(legacy_eval)
        if ffi is None or legacy is None:
            # skip this day if missing either metric
            continue
        rows.append({
            "day": int(day),
            "FFI": float(ffi),
            "LegacyAcc": float(legacy),
            # Optional columns if missing; we can add NaNs or zeros
            "train_seconds": float("nan"),
            "n_train": float("nan"),
            "buffer_size": float("nan"),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    return df


def load_summary(path: Path) -> pd.DataFrame:
    """Load a summary CSV and minimally validate."""
    df = pd.read_csv(path)
    # Ensure required columns exist
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    # types
    df["day"] = df["day"].astype(int)
    df["FFI"] = df["FFI"].astype(float)
    df["LegacyAcc"] = df["LegacyAcc"].astype(float)
    return df


def discover_run_dirs(artifacts_root: Path) -> List[Path]:
    """
    Discover run directories under artifacts.
    We consider any directory that contains either:
      - a summary.csv, OR
      - ffi/day_*/eval.json AND legacy/day_*/eval.json
    """
    candidates = set()

    # 1) Any folder that already has a summary.csv
    for p in artifacts_root.rglob("summary.csv"):
        if p.is_file():
            candidates.add(p.parent)

    # 2) Any folder with evals (ffi & legacy)
    for p in artifacts_root.rglob("ffi"):
        run_dir = p.parent
        if (run_dir / "legacy").exists():
            candidates.add(run_dir)

    # Only keep things that look like rq2_* runs or adapters’ parents
    # (Relaxed: keep them all; labeling will give context)
    return sorted(candidates)


def collect_runs(artifacts_root: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (label, summary_csv_path).
    If a run is missing a valid summary.csv, rebuild from evals into <run_dir>/summary.csv.
    """
    runs: List[Tuple[str, Path]] = []
    for run_dir in discover_run_dirs(artifacts_root):
        label = pretty_label(run_dir.name)
        summary_csv = run_dir / "summary.csv"

        if summary_csv.exists():
            # Check schema; if bad, attempt rebuild
            try:
                _ = load_summary(summary_csv)
            except Exception:
                rebuilt = rebuild_summary_from_evals(run_dir)
                if rebuilt is not None:
                    rebuilt.to_csv(summary_csv, index=False)
                else:
                    # If we cannot rebuild, skip this run
                    continue
        else:
            # Try to build from evals
            rebuilt = rebuild_summary_from_evals(run_dir)
            if rebuilt is not None:
                rebuilt.to_csv(summary_csv, index=False)
            else:
                # nothing to do for this run
                continue

        # Final verification
        try:
            _ = load_summary(summary_csv)
            runs.append((label, summary_csv))
        except Exception:
            # Skip if still invalid
            continue

    return runs


def filtered_plot(df: pd.DataFrame, outdir: Path, predicate, postfix: str):
    sub = df[df["RegimenLabel"].apply(predicate)]
    if sub.empty:
        return

    # FFI
    plt.figure()
    for k, g in sub.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["FFI"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("FFI (Exact Match)")
    plt.title(f"Freshness Over Days {postfix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / f"ffi_over_days_{postfix}.png", dpi=180, bbox_inches="tight")

    # Legacy
    plt.figure()
    for k, g in sub.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["LegacyAcc"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Legacy Accuracy (Exact Match)")
    plt.title(f"Legacy Retention Over Days {postfix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / f"legacy_over_days_{postfix}.png", dpi=180, bbox_inches="tight")


def main(args):
    artifacts_root = Path(args.artifacts_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_map: List[Tuple[str, Path]] = []
    if args.paths:
        # format: "Label1=path/to/summary.csv,Label2=path2,..."
        for item in args.paths.split(","):
            if not item.strip():
                continue
            label, path = item.split("=", 1)
            run_map.append((label.strip(), Path(path.strip())))
    else:
        run_map = collect_runs(artifacts_root)

    if not run_map:
        print(
            "No runs found.\n"
            "I tried recursively under artifacts/ to find either:\n"
            " - summary.csv files, or\n"
            " - ffi/day_*/eval.json + legacy/day_*/eval.json folders to rebuild summaries.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = []
    for label, path in run_map:
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        try:
            df = load_summary(path)
        except Exception as e:
            print(f"[warn] skipping {path} due to: {e}", file=sys.stderr)
            continue
        df["RegimenLabel"] = label
        frames.append(df)

    if not frames:
        raise SystemExit("Found candidate runs, but none yielded a valid summary.")

    df = pd.concat(frames, ignore_index=True)

    # Overall FFI over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["FFI"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("FFI (Exact Match)")
    plt.title("Freshness Over Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "ffi_over_days.png", dpi=180, bbox_inches="tight")

    # Overall Legacy over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["LegacyAcc"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Legacy Accuracy (Exact Match)")
    plt.title("Legacy Retention Over Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "legacy_over_days.png", dpi=180, bbox_inches="tight")

    # Optional filtered overlays (smoke-only / full-only)
    has_smoke = any("(smoke)" in s for s in df["RegimenLabel"].unique())
    has_full = any("(full)" in s for s in df["RegimenLabel"].unique())
    if has_smoke:
        filtered_plot(df, outdir, lambda s: "(smoke)" in s, "smoke_only")
    if has_full:
        filtered_plot(df, outdir, lambda s: "(full)" in s, "full_only")

    # Final day comparison table (per label)
    last = (
        df.sort_values("day")
          .groupby("RegimenLabel", as_index=False)
          .tail(1)[["RegimenLabel", "FFI", "LegacyAcc"]]
          .reset_index(drop=True)
    )
    last.to_csv(outdir / "final_day_summary.csv", index=False)

    # Save a combined CSV for convenience
    df.to_csv(outdir / "all_runs_concat.csv", index=False)

    print("Analyzed runs:")
    for label, path in run_map:
        print(f" - {label}: {path}")
    print("Wrote figures to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", default="artifacts", help="Root folder to search")
    ap.add_argument(
        "--paths",
        default="",
        help='Optional explicit mapping: "LoRA-only (full)=artifacts/rq2_lora_only/summary.csv,Replay-only (smoke)=artifacts/rq2_replay_only_smoke/summary.csv".',
    )
    ap.add_argument("--out_dir", default="artifacts/figures")
    args = ap.parse_args()
    main(args)
