"""Sweep topk_val for gqa_quest_sparse_attention with fp8 KV cache.

Mirrors the OmniServe LServe `dynamic_sparse_token_budget` sweep so the
two systems can be compared on AIME24 under matched sparsity budgets.

Each topk_val launches verify_algo.py in a fresh subprocess so the SGLang
engine is fully torn down between runs (no leaked GPU memory). Per-run
summary JSONs are written to `<summary_dir>/topk_<val>/` and a final
aggregated table is written to `<summary_dir>/sweep_summary.json`.

Usage (run from the vortex_torch repo root):
    python examples/compare_omni.py
    python examples/compare_omni.py --topk-vals 29 61 93 --trials 4
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_TOPK_VALS = [29, 61, 93, 125, 157, 189, 221, 253]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topk-vals", type=int, nargs="+", default=DEFAULT_TOPK_VALS,
                   help="Sweep of vortex_topk_val values.")
    p.add_argument("--topk-ratio", type=float, default=0.0)
    p.add_argument("--vortex-module-name", type=str, default="gqa_quest_sparse_attention")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--data-path", type=str,
                   default="examples/aime24_llama8b.jsonl")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--kv-cache-dtype", type=str, default="fp8_e4m3")
    p.add_argument("--summary-dir", type=str, default="summary_compare_omni")
    p.add_argument("--verify-script", type=str, default="examples/verify_algo.py")
    return p.parse_args()


def latest_summary(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    args = parse_args()
    os.makedirs(args.summary_dir, exist_ok=True)

    aggregated = []
    for topk_val in args.topk_vals:
        run_dir = os.path.join(args.summary_dir, f"topk_{topk_val}")
        cmd = [
            sys.executable, args.verify_script,
            "--topk-val", str(topk_val),
            "--topk-ratio", str(args.topk_ratio),
            "--vortex-module-name", args.vortex_module_name,
            "--model-name", args.model_name,
            "--data-path", args.data_path,
            "--trials", str(args.trials),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--summary-dir", run_dir,
        ]
        print(f"[compare_omni] topk_val={topk_val} -> {run_dir}", flush=True)
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary_path = latest_summary(run_dir)
        if summary_path is None:
            print(f"[compare_omni] WARN: no summary written to {run_dir}", flush=True)
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        aggregated.append({"topk_val": topk_val, "summary_path": summary_path, "summary": summary})

    out_path = os.path.join(
        args.summary_dir,
        f"sweep_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "runs": aggregated,
        }, f, ensure_ascii=False, indent=2)
    print(f"[compare_omni] aggregated sweep written to {out_path}")

    print("\ntopk_val | mean@T  | pass@T  | tokens   | e2e(s)  | tput(tok/s)")
    print("-" * 64)
    for r in aggregated:
        s = r["summary"]
        trials = s.get("args", {}).get("trials", "?")
        mean = s.get(f"mean@{trials}", float("nan"))
        passk = s.get(f"pass@{trials}", float("nan"))
        print(f"{r['topk_val']:>7} | {mean:7.4f} | {passk:7.4f} | "
              f"{s.get('total_tokens',0):>8} | {s.get('e2e_time',0):7.2f} | "
              f"{s.get('throughput',0):10.2f}")


if __name__ == "__main__":
    main()
