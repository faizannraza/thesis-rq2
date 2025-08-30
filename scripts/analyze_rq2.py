# scripts/analyze_rq2.py
import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


def pretty_label(dir_name: str) -> str:
    name = dir_name.lower()
    if "lora_only" in name:
        base = "LoRA-only"
    elif "replay_only" in name:
        base = "Replay-only"
    elif "hybrid" in name:
        base = "Hybrid"
    else:
        base = dir_name
    if "smoke" in name:
        base += " (smoke)"
    return base


def discover_runs(artifacts_root: Path) -> list[tuple[str, Path]]:
    runs = []
    for p in artifacts_root.glob("rq2_*/summary.csv"):
        label = pretty_label(p.parent.name)
        runs.append((label, p))
    return runs


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal schema check
    for col in ["day", "FFI", "LegacyAcc"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
    return df


def main(args):
    artifacts_root = Path(args.artifacts_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # If explicit paths were provided, use them; else auto-discover
    run_map = []
    if args.paths:
        # format: "Label1=path/to/summary.csv,Label2=path2,..."
        for item in args.paths.split(","):
            if not item.strip():
                continue
            label, path = item.split("=", 1)
            run_map.append((label.strip(), Path(path.strip())))
    else:
        run_map = discover_runs(artifacts_root)

    if not run_map:
        print(
            "No runs found.\n"
            "Expected to find at least one summary.csv under artifacts/rq2_*/summary.csv.\n"
            "Example smoke paths:\n"
            "  artifacts/rq2_lora_only_smoke/summary.csv\n"
            "  artifacts/rq2_replay_only_smoke/summary.csv\n"
            "  artifacts/rq2_hybrid_smoke/summary.csv\n",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = []
    for label, path in run_map:
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        df = load_summary(path)
        df["RegimenLabel"] = label
        frames.append(df)

    if not frames:
        raise SystemExit("Found run entries, but none of the summary.csv files exist.")

    df = pd.concat(frames, ignore_index=True)

    # FFI over days
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

    # Legacy over days
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

    # Final day comparison table
    last = (
        df.sort_values("day").groupby("RegimenLabel").tail(1)[
            ["RegimenLabel", "FFI", "LegacyAcc"]
        ]
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
        help='Optional explicit mapping: "LoRA-only=artifacts/rq2_lora_only_smoke/summary.csv,Replay-only=...".',
    )
    ap.add_argument("--out_dir", default="artifacts/figures")
    args = ap.parse_args()
    main(args)
