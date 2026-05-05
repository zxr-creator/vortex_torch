"""Kernel-level benchmark for the three top-k variants in vortex_torch_C:
  - topk_output         (csrc/topk.cu)
  - topk_output_v2      (csrc/topk_v2.cu)
  - approx_topk_output  (csrc/approx_topk.cu)

For each model architecture the script reads num_key_value_heads from the HF
config and uses (batch_size * num_kv_heads) as the effective number of rows
the kernel parallelizes over. Per-row dense block count is derived from
input_len / block_size, matching the page layout used in compare_omni.sh.

This is a kernel-only benchmark: no model weights are loaded.
"""

import argparse
import json
import os
import statistics
from typing import Callable, Dict, List

import torch
from transformers import AutoConfig

from vortex_torch_C import (
    topk_output,
    topk_output_v2,
    approx_topk_output,
    approx_topk_output_remap,
    topk_output_v2_remap,
    topk_output_sglang_ori,
)

# sglang_topk.cu hard-codes TopK at compile time. The current build sets it to
# this value; configs whose topk_val != SGLANG_ORI_TOPK are skipped for that
# kernel. Bumping this means rebuilding sglang_topk.cu.
SGLANG_ORI_TOPK = 32

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# Curated remap candidates for the *_remap kernel autotune. Each tuple is
# (tag, mapping_mode, mapping_power); modes match TopKMappingMode in
# csrc/topk_mapping.cuh. NONE is the identity baseline. The other six
# cover spread, log/asinh-style compression, and top-region amplifiers.
REMAP_CANDIDATES = [
    ("NONE",           0, 0.0),
    ("POWER@0.5",      3, 0.5),
    ("POWER@2.0",      3, 2.0),
    ("ASINH@1.0",      6, 1.0),
    ("LOG1P@1.0",      7, 1.0),
    ("SHIFT_POW2@0",  15, 0.0),
    ("HALF_SQUARE@0", 18, 0.0),
]

# A remap candidate is accepted if its recall@topk_val is within this slack
# of the NONE baseline's recall. Latency then breaks ties.
RECALL_FLOOR_SLACK = 0.005

# Lightweight autotune iters — full timing happens after the chosen config
# is locked in.
AUTOTUNE_WARMUP = 5
AUTOTUNE_ITERS  = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True,
                        help='HF model id, e.g. "Qwen/Qwen3-1.7B".')
    parser.add_argument("--model-label", default=None,
                        help="Short label used in output records (default: model-name with / replaced).")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True,
                        help="Per-request context length in tokens.")
    parser.add_argument("--input-label", default=None,
                        help="Short label for input length (e.g. 4k); defaults to str(input_len).")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Tokens per KV block (default: 16, matches compare_omni.sh).")
    parser.add_argument("--topk-vals", default="29,61,125,253",
                        help="Comma-separated topk values to sweep "
                             "(default: subset of compare_omni.sh sweep).")
    parser.add_argument("--tolerate-ratios", default="0.0,0.05,0.1",
                        help="Comma-separated tolerate_ratio values for approx_topk_output.")
    parser.add_argument("--reserved-bos", type=int, default=1,
                        help="Blocks reserved at the start (default: 1, matches sparse-attn configs).")
    parser.add_argument("--reserved-eos", type=int, default=2,
                        help="Blocks reserved at the end (default: 2, matches sparse-attn configs).")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Score tensor dtype (default: bfloat16, matches Q/K dtype).")
    parser.add_argument("--distribution", default="normal",
                        choices=["uniform", "normal", "lognormal", "real", "bimodal"],
                        help="Score-tensor distribution. 'real'/'lognormal' are "
                             "aliases for the heavy-tailed lognormal proxy. "
                             "'bimodal' is a sparse-spike mixture (95%% low-noise "
                             "+ 5%% high-spike).")
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", default=None,
                        help="Append one JSON record per measurement to this path.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_scores(distribution: str, shape, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Allocate the score tensor used by the top-k kernels under one of four distributions.

    Sampling is done in float32 for numerical stability, then cast to `dtype`.
    """
    if distribution == "uniform":
        x = torch.rand(shape, device=device, dtype=torch.float32).mul_(2.0).sub_(1.0)
    elif distribution == "normal":
        x = torch.randn(shape, device=device, dtype=torch.float32)
    elif distribution == "real" or distribution == "lognormal":
        # Heavy-tailed lognormal proxy for post-softmax attention block scores.
        x = torch.randn(shape, device=device, dtype=torch.float32).exp_()
    elif distribution == "bimodal":
        # 95% low-noise background + 5% high-spike component.
        bg = torch.randn(shape, device=device, dtype=torch.float32).mul_(0.1)
        spike = torch.randn(shape, device=device, dtype=torch.float32).mul_(0.5).add_(5.0)
        mask = torch.rand(shape, device=device, dtype=torch.float32) < 0.05
        x = torch.where(mask, spike, bg)
    else:
        raise ValueError(f"unknown distribution: {distribution!r}")
    return x.to(dtype)


def time_kernel(fn: Callable[[], None], num_warmup: int, num_iters: int) -> Dict[str, float]:
    """Returns mean / p50 / p95 / min latency in milliseconds for `fn`."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times_ms)
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "p50_ms":  float(times_ms[n // 2]),
        "p95_ms":  float(times_ms[min(n - 1, max(0, int(n * 0.95) - 1))]),
        "min_ms":  float(times_ms[0]),
    }


def compute_recall_at_ks(
    x: torch.Tensor,
    selected_global: torch.Tensor,
    eff_batch_size: int,
    blocks_per_row: int,
    k_values: List[int],
    reserved_bos: int,
    reserved_eos: int,
) -> Dict[int, float]:
    """Mean recall@k of `selected_global` against torch.topk on the candidate
    region of `x` (per row, excluding reserved BOS/EOS blocks), for each k in
    `k_values`.

    `x` has shape (eff_batch_size * blocks_per_row, 1, 1). `selected_global`
    has shape (eff_batch_size, S) where S is per_row_sparse for the dense
    kernels (BOS + topk + EOS) or topk_val for the sglang kernel.

    Recall@k is |kernel_selected ∩ ground_truth_top_k| / k. Reserved BOS/EOS
    blocks in the kernel output don't penalize recall; they simply don't
    intersect the ground-truth set (which lives in the candidate region).
    Returns NaN for any k <= 0 or k larger than the candidate region.
    """
    cand_len = blocks_per_row - reserved_bos - reserved_eos
    if cand_len <= 0:
        return {k: float("nan") for k in k_values}

    x_2d = x.view(eff_batch_size, blocks_per_row).float()
    cand = x_2d[:, reserved_bos:blocks_per_row - reserved_eos]
    row_offsets = (
        torch.arange(eff_batch_size, device=x.device, dtype=torch.long)
        * blocks_per_row
    ).unsqueeze(1)

    k_max = min(max(k_values), cand_len)
    gt_local_max = cand.topk(k=k_max, dim=1).indices  # (B, k_max)
    gt_global_max = gt_local_max.long() + reserved_bos + row_offsets

    sel = selected_global.long()  # (B, S)
    out: Dict[int, float] = {}
    for k in k_values:
        if k <= 0 or k > cand_len:
            out[k] = float("nan")
            continue
        gt_k = gt_global_max[:, :k]  # (B, k)
        matches = (gt_k.unsqueeze(2) == sel.unsqueeze(1)).any(dim=2)
        out[k] = float(matches.float().mean().item())
    return out


def autotune_remap(
    kernel_factory: Callable[[int, float], Callable[[], None]],
    out_buf: torch.Tensor,
    out_shape: tuple,
    x: torch.Tensor,
    eff_batch_size: int,
    blocks_per_row: int,
    topk_val: int,
    reserved_bos: int,
    reserved_eos: int,
) -> Dict[str, float]:
    """Time each REMAP_CANDIDATES entry on the given kernel and pick the
    fastest one whose recall@topk_val is within RECALL_FLOOR_SLACK of the
    NONE baseline. Falls back to NONE if no candidate clears the floor.

    `kernel_factory(mode, power)` returns a callable that runs the kernel.
    Returns a dict with keys: tag, mapping_mode, mapping_power,
    baseline_recall_at_topk.
    """
    candidate_results = []  # list of (tag, mode, power, mean_ms, recall)
    for tag, mode, power in REMAP_CANDIDATES:
        fn = kernel_factory(mode, power)
        try:
            stats = time_kernel(fn, AUTOTUNE_WARMUP, AUTOTUNE_ITERS)
        except RuntimeError:
            continue
        try:
            out_buf.zero_()
            fn()
            torch.cuda.synchronize()
            selected = out_buf.view(*out_shape)
            recalls = compute_recall_at_ks(
                x, selected, eff_batch_size, blocks_per_row,
                [topk_val], reserved_bos, reserved_eos,
            )
            recall = recalls.get(topk_val, float("nan"))
        except RuntimeError:
            recall = float("nan")
        candidate_results.append((tag, mode, power, stats["mean_ms"], recall))

    none_entry = next((r for r in candidate_results if r[0] == "NONE"), None)
    baseline_recall = none_entry[4] if none_entry is not None else float("nan")

    if baseline_recall != baseline_recall:  # NaN baseline → bail to NONE
        eligible = []
    else:
        eligible = [r for r in candidate_results
                    if r[4] == r[4] and r[4] >= baseline_recall - RECALL_FLOOR_SLACK]

    if not eligible:
        return {"tag": "NONE", "mapping_mode": 0, "mapping_power": 0.0,
                "baseline_recall_at_topk": baseline_recall}

    best = min(eligible, key=lambda r: r[3])
    return {"tag": best[0], "mapping_mode": best[1], "mapping_power": best[2],
            "baseline_recall_at_topk": baseline_recall}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = AutoConfig.from_pretrained(args.model_name)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    model_label = args.model_label or args.model_name.replace("/", "_")
    input_label = args.input_label or str(args.input_len)

    eff_batch_size = args.batch_size * num_kv_heads
    blocks_per_row = (args.input_len + args.block_size - 1) // args.block_size
    total_dense_blocks = eff_batch_size * blocks_per_row

    dtype = {"bfloat16": torch.bfloat16,
             "float16":  torch.float16,
             "float32":  torch.float32}[args.dtype]

    # ----- Build the inputs that are constant across the topk_val sweep -----
    x = make_scores(args.distribution, (total_dense_blocks, 1, 1), dtype, args.device)
    dense_kv_indptr = (
        torch.arange(eff_batch_size + 1, dtype=torch.int32, device=args.device)
        * blocks_per_row
    )
    dense_kv_indices = torch.arange(total_dense_blocks, dtype=torch.int32, device=args.device)

    topk_vals = [int(s) for s in args.topk_vals.split(",") if s.strip()]
    tolerate_ratios = [float(s) for s in args.tolerate_ratios.split(",") if s.strip()]

    records: List[dict] = []

    for topk_val in topk_vals:
        # Per-row sparse block count: bounded above by what's actually present.
        per_row_sparse = min(
            blocks_per_row,
            topk_val + args.reserved_bos + args.reserved_eos,
        )
        sparse_kv_indptr = (
            torch.arange(eff_batch_size + 1, dtype=torch.int32, device=args.device)
            * per_row_sparse
        )
        sparse_kv_indices = torch.zeros(
            (eff_batch_size * per_row_sparse, 1, 1),
            dtype=torch.int32, device=args.device,
        )

        common_args = (
            x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, args.reserved_bos, args.reserved_eos, blocks_per_row,
        )

        # Each kernel entry: (name, fn, tolerate_ratio, output_buf,
        # output_view_shape, extra_fields). extra_fields is a dict (or None)
        # of additional record fields used by the *_remap autotuned kernels
        # to surface their chosen mapping_mode / mapping_power.
        kernels: List[tuple] = [
            ("topk_output",    lambda: topk_output(*common_args),
             None, sparse_kv_indices, (eff_batch_size, per_row_sparse), None),
            ("topk_output_v2", lambda: topk_output_v2(*common_args),
             None, sparse_kv_indices, (eff_batch_size, per_row_sparse), None),
        ]
        if topk_val == SGLANG_ORI_TOPK:
            sglang_indices_out = torch.zeros(
                (eff_batch_size, topk_val), dtype=torch.int32, device=args.device,
            )
            sglang_args = (
                x, dense_kv_indptr, sglang_indices_out,
                eff_batch_size, topk_val,
                args.reserved_bos, args.reserved_eos, blocks_per_row,
            )
            kernels.append((
                "topk_output_sglang_ori",
                lambda: topk_output_sglang_ori(*sglang_args),
                None, sglang_indices_out, (eff_batch_size, topk_val), None,
            ))
        for tr in tolerate_ratios:
            tr_local = tr  # capture for closure
            kernels.append((
                f"approx_topk_output@{tr_local:g}",
                lambda tr=tr_local: approx_topk_output(*common_args, tr),
                tr_local, sparse_kv_indices, (eff_batch_size, per_row_sparse),
                None,
            ))

        # Autotune the two *_remap kernels per (model, distribution, topk)
        # config. For each, time every REMAP_CANDIDATES entry, then pick the
        # fastest with recall@topk_val ≥ NONE_recall - RECALL_FLOOR_SLACK.
        # The chosen mapping is then run through the full timed measurement
        # alongside the unmapped kernels for an apples-to-apples comparison.
        approx_remap_factory = lambda mode, power: (
            lambda: approx_topk_output_remap(*common_args, 0.0, mode, power)
        )
        topkv2_remap_factory = lambda mode, power: (
            lambda: topk_output_v2_remap(*common_args, mode, power)
        )

        approx_chosen = autotune_remap(
            approx_remap_factory, sparse_kv_indices,
            (eff_batch_size, per_row_sparse),
            x, eff_batch_size, blocks_per_row, topk_val,
            args.reserved_bos, args.reserved_eos,
        )
        topkv2_chosen = autotune_remap(
            topkv2_remap_factory, sparse_kv_indices,
            (eff_batch_size, per_row_sparse),
            x, eff_batch_size, blocks_per_row, topk_val,
            args.reserved_bos, args.reserved_eos,
        )

        kernels.append((
            f"approx_topk_output_remap@{approx_chosen['tag']}",
            approx_remap_factory(approx_chosen["mapping_mode"], approx_chosen["mapping_power"]),
            None, sparse_kv_indices, (eff_batch_size, per_row_sparse),
            {"mapping_mode":  approx_chosen["mapping_mode"],
             "mapping_power": approx_chosen["mapping_power"],
             "mapping_tag":   approx_chosen["tag"],
             "autotune_baseline_recall_at_topk": approx_chosen["baseline_recall_at_topk"]},
        ))
        kernels.append((
            f"topk_output_v2_remap@{topkv2_chosen['tag']}",
            topkv2_remap_factory(topkv2_chosen["mapping_mode"], topkv2_chosen["mapping_power"]),
            None, sparse_kv_indices, (eff_batch_size, per_row_sparse),
            {"mapping_mode":  topkv2_chosen["mapping_mode"],
             "mapping_power": topkv2_chosen["mapping_power"],
             "mapping_tag":   topkv2_chosen["tag"],
             "autotune_baseline_recall_at_topk": topkv2_chosen["baseline_recall_at_topk"]},
        ))

        for kernel_name, fn, tr, out_buf, out_shape, extra_fields in kernels:
            try:
                stats = time_kernel(fn, args.num_warmup, args.num_iters)
            except RuntimeError as e:
                print(
                    f"[{model_label}|bs={args.batch_size}|in={input_label}|"
                    f"topk={topk_val}|dist={args.distribution}|{kernel_name}] "
                    f"SKIPPED: {e}"
                )
                continue

            # One additional (untimed) call to capture this kernel's selection
            # for recall@k. Different kernels share sparse_kv_indices, so we
            # must re-run before reading the buffer.
            recall_k_values = sorted({32, 64, 128, topk_val})
            try:
                out_buf.zero_()
                fn()
                torch.cuda.synchronize()
                selected = out_buf.view(*out_shape)
                recalls = compute_recall_at_ks(
                    x, selected,
                    eff_batch_size, blocks_per_row,
                    recall_k_values, args.reserved_bos, args.reserved_eos,
                )
            except RuntimeError as e:
                print(
                    f"[{model_label}|bs={args.batch_size}|in={input_label}|"
                    f"topk={topk_val}|dist={args.distribution}|{kernel_name}] "
                    f"recall computation FAILED: {e}"
                )
                recalls = {k: float("nan") for k in recall_k_values}

            record = {
                "model": model_label,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "input_label": input_label,
                "block_size": args.block_size,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "eff_batch_size": eff_batch_size,
                "blocks_per_row": blocks_per_row,
                "total_dense_blocks": total_dense_blocks,
                "topk_val": topk_val,
                "per_row_sparse": per_row_sparse,
                "reserved_bos": args.reserved_bos,
                "reserved_eos": args.reserved_eos,
                "kernel": kernel_name,
                "tolerate_ratio": tr,
                "dtype": args.dtype,
                "distribution": args.distribution,
                "num_warmup": args.num_warmup,
                "num_iters": args.num_iters,
                "recall_at_32":   recalls.get(32,       float("nan")),
                "recall_at_64":   recalls.get(64,       float("nan")),
                "recall_at_128":  recalls.get(128,      float("nan")),
                "recall_at_topk": recalls.get(topk_val, float("nan")),
                **stats,
            }
            if extra_fields:
                record.update(extra_fields)
            records.append(record)

            def _fmt(v: float) -> str:
                return f"{v:.4f}" if v == v else "nan"  # NaN-safe
            print(
                f"[{model_label}|bs={args.batch_size}|in={input_label}|"
                f"topk={topk_val}|dist={args.distribution}|{kernel_name}] "
                f"mean={stats['mean_ms']:.4f}ms  p50={stats['p50_ms']:.4f}ms  "
                f"p95={stats['p95_ms']:.4f}ms  min={stats['min_ms']:.4f}ms  "
                f"R@32={_fmt(recalls.get(32, float('nan')))}  "
                f"R@64={_fmt(recalls.get(64, float('nan')))}  "
                f"R@128={_fmt(recalls.get(128, float('nan')))}  "
                f"R@topk({topk_val})={_fmt(recalls.get(topk_val, float('nan')))}"
            )

    if args.output_jsonl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
        with open(args.output_jsonl, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
