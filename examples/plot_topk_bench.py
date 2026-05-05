"""Plot filtered top-k kernel benchmark results.

Requested settings only:
  1. input_len = 2048, topk_val in {32, 64, 128, 256}
  2. input_len in {32768, 65536, 131072}, topk_val = 2048

Kernel names shown in the figure:
  - topk_output        -> Sort TopK
  - topk_v2           -> Radix TopK
  - approx_topk       -> Approx Radix TopK
  - topk_v2 + remap   -> Radix TopK + Remap
  - approx_topk+remap -> Approx Radix TopK + Remap

Notes:
  - topk_sglang_ori / topk_output_sglang_ori are intentionally excluded.
  - Error bars are intentionally disabled: no yerr / capsize, so bars do not
    contain the vertical "I" markers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = Path("/data/datasets/xinrui/topk_bench_20260505_075252.csv")
DEFAULT_OUT = Path("/data/datasets/xinrui/topk_kernels.pdf")

KEY = [
    "model",
    "batch_size",
    "distribution",
    "input_len",
    "blocks_per_row",
    "topk_val",
]

SMALL_INPUT_LEN = 2048
SMALL_TOPK_VALS = [32, 64, 128, 256]
LARGE_INPUT_LENS = [32768, 65536, 131072]
LARGE_TOPK_VAL = 2048

METHOD_ORDER = [
    "radix_topk",
    "approx_radix_topk",
    "radix_topk_remap",
    "approx_radix_topk_remap",
]
SPEEDUP_METHOD_ORDER = [m for m in METHOD_ORDER if m != "radix_topk"]

METHOD_LABEL = {
    "radix_topk": "Radix TopK",
    "approx_radix_topk": "Approx Radix TopK",
    "radix_topk_remap": "Radix TopK + Remap",
    "approx_radix_topk_remap": "Approx Radix TopK + Remap",
}

# Soft, colorblind-friendly conference-style palette.
METHOD_COLOR = {
    "radix_topk": "#66C2A5",                # soft teal
    "approx_radix_topk": "#FC8D62",         # soft orange
    "radix_topk_remap": "#8DA0CB",          # soft blue-purple
    "approx_radix_topk_remap": "#E78AC3",   # soft pink
}

KERNEL_ALIAS = {
    # Baseline: full sort.
    "topk_output": "sort_topk",
    "sort_topk": "sort_topk",
    # Radix TopK.
    "topk_v2": "radix_topk",
    "topk_output_v2": "radix_topk",
    "radix_topk": "radix_topk",
    # Approx Radix TopK.
    "approx_topk": "approx_radix_topk",
    "approx_topk_output": "approx_radix_topk",
    "approx_radix_topk": "approx_radix_topk",
    # Radix TopK + Remap.
    "topk_v2_remap": "radix_topk_remap",
    "topk_output_v2_remap": "radix_topk_remap",
    "radix_topk_remap": "radix_topk_remap",
    # Approx Radix TopK + Remap.
    "approx_topk_remap": "approx_radix_topk_remap",
    "approx_topk_output_remap": "approx_radix_topk_remap",
    "approx_radix_topk_remap": "approx_radix_topk_remap",
}

EXCLUDED_KERNELS = {
    "topk_sglang_ori",
    "topk_output_sglang_ori",
    "sglang_ori",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def kernel_base_name(kernel: str) -> str:
    """Strip tuning suffixes such as '@alpha=...' or '@mapping=...'."""
    return str(kernel).split("@", 1)[0]


def canonical_method(kernel_base: str) -> str | None:
    if kernel_base in EXCLUDED_KERNELS:
        return None
    return KERNEL_ALIAS.get(kernel_base)


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"kernel", "mean_ms", "input_len", "topk_val"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df["kernel_base"] = df["kernel"].map(kernel_base_name)
    df["method"] = df["kernel_base"].map(canonical_method)

    # Drop SGLang and any kernel not in the requested method list.
    df = df[df["method"].isin(METHOD_ORDER)].copy()

    # Keep only the requested benchmark settings.
    small = (df["input_len"].eq(SMALL_INPUT_LEN) &
             df["topk_val"].isin(SMALL_TOPK_VALS))
    large = (df["input_len"].isin(LARGE_INPUT_LENS) &
             df["topk_val"].eq(LARGE_TOPK_VAL))
    df = df[small | large].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after filtering. Expected settings are: "
            "2048 -> {32,64,128,256}, and {32768,65536,131072} -> 2048."
        )

    return df


def best_latency_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Use the minimum latency per kernel family for each benchmark config."""
    group_cols = [c for c in KEY if c in df.columns] + ["method"]
    idx = df.groupby(group_cols, dropna=False)["mean_ms"].idxmin()
    best = df.loc[idx].copy()
    return best


def aggregate_latency(best: pd.DataFrame, configs: Iterable[tuple[int, int]]) -> pd.DataFrame:
    """Aggregate latency over model / batch / distribution for each setting."""
    rows = []
    for input_len, topk_val in configs:
        for method in METHOD_ORDER:
            s = best[
                best["input_len"].eq(input_len)
                & best["topk_val"].eq(topk_val)
                & best["method"].eq(method)
            ]
            rows.append({
                "input_len": input_len,
                "topk_val": topk_val,
                "method": method,
                "mean_ms": s["mean_ms"].mean() if len(s) else np.nan,
                "num_points": int(len(s)),
            })
    return pd.DataFrame(rows)


def add_speedup(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    baseline = (
        agg[agg["method"].eq("sort_topk")]
        .set_index(["input_len", "topk_val"])["mean_ms"]
    )

    def _speedup(row: pd.Series) -> float:
        key = (row["input_len"], row["topk_val"])
        base = baseline.get(key, np.nan)
        if pd.isna(base) or pd.isna(row["mean_ms"]) or row["mean_ms"] == 0:
            return np.nan
        return float(base / row["mean_ms"])

    agg["speedup_vs_sort"] = agg.apply(_speedup, axis=1)
    return agg


def grouped_bar(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    configs: list[tuple[int, int]],
    methods: list[str],
    value_col: str,
    ylabel: str,
    title: str,
    xlabels: list[str],
    ylim_bottom: float | None = None,
) -> None:
    x = np.arange(len(configs))
    width = min(0.78 / max(len(methods), 1), 0.18)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, method in zip(offsets, methods):
        vals = []
        for input_len, topk_val in configs:
            row = data[
                data["input_len"].eq(input_len)
                & data["topk_val"].eq(topk_val)
                & data["method"].eq(method)
            ]
            vals.append(row[value_col].iloc[0] if len(row) else np.nan)

        # No yerr / capsize here: this intentionally removes the vertical "I".
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=METHOD_LABEL[method],
            color=METHOD_COLOR[method],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.96,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)


def plot_combined(best: pd.DataFrame, out_path: Path) -> None:
    small_configs = [(SMALL_INPUT_LEN, k) for k in SMALL_TOPK_VALS]
    large_configs = [(n, LARGE_TOPK_VAL) for n in LARGE_INPUT_LENS]

    small = add_speedup(aggregate_latency(best, small_configs))
    large = add_speedup(aggregate_latency(best, large_configs))

    # Convert ms -> us for latency panels.
    small["latency_us"] = small["mean_ms"] * 1000.0
    large["latency_us"] = large["mean_ms"] * 1000.0

    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.3))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    grouped_bar(
        ax=ax_a,
        data=small,
        configs=small_configs,
        methods=METHOD_ORDER,
        value_col="latency_us",
        ylabel=r"Latency ($\mu$s)",
        title=r"(a) Latency: 2K input",
        xlabels=[f"2K$\\rightarrow${k}" for k in SMALL_TOPK_VALS],
        ylim_bottom=0.0,
    )

    grouped_bar(
        ax=ax_b,
        data=small,
        configs=small_configs,
        methods=SPEEDUP_METHOD_ORDER,
        value_col="speedup_vs_sort",
        ylabel=r"Speedup vs Sort TopK ($\times$)",
        title=r"(b) Speedup: 2K input",
        xlabels=[f"2K$\\rightarrow${k}" for k in SMALL_TOPK_VALS],
        ylim_bottom=0.0,
    )
    ax_b.axhline(1.0, color="#777777", lw=0.8, ls="--", zorder=0)

    grouped_bar(
        ax=ax_c,
        data=large,
        configs=large_configs,
        methods=METHOD_ORDER,
        value_col="latency_us",
        ylabel=r"Latency ($\mu$s)",
        title=r"(c) Latency: top-2048",
        xlabels=[f"{n // 1024}K$\\rightarrow$2048" for n in LARGE_INPUT_LENS],
        ylim_bottom=0.0,
    )

    grouped_bar(
        ax=ax_d,
        data=large,
        configs=large_configs,
        methods=SPEEDUP_METHOD_ORDER,
        value_col="speedup_vs_sort",
        ylabel=r"Speedup vs Sort TopK ($\times$)",
        title=r"(d) Speedup: top-2048",
        xlabels=[f"{n // 1024}K$\\rightarrow$2048" for n in LARGE_INPUT_LENS],
        ylim_bottom=0.0,
    )
    ax_d.axhline(1.0, color="#777777", lw=0.8, ls="--", zorder=0)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        framealpha=0.94,
        edgecolor="#DDDDDD",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.0,
        handlelength=1.4,
    )

    for ax in axes.flatten():
        ax.margins(x=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def print_summary(best: pd.DataFrame) -> None:
    configs = [(SMALL_INPUT_LEN, k) for k in SMALL_TOPK_VALS]
    configs += [(n, LARGE_TOPK_VAL) for n in LARGE_INPUT_LENS]
    agg = add_speedup(aggregate_latency(best, configs))

    print("\n=== Filtered settings ===")
    for input_len, topk_val in configs:
        sub = agg[agg["input_len"].eq(input_len) & agg["topk_val"].eq(topk_val)]
        available = sub[sub["num_points"].gt(0)]["method"].map(METHOD_LABEL).tolist()
        missing = sub[sub["num_points"].eq(0)]["method"].map(METHOD_LABEL).tolist()
        print(f"input_len={input_len}, topk={topk_val}: available={available}")
        if missing:
            print(f"  missing={missing}")

    print("\n=== Mean speedup vs Sort TopK ===")
    table = agg[agg["method"].isin(SPEEDUP_METHOD_ORDER)].pivot_table(
        index=["input_len", "topk_val"],
        columns="method",
        values="speedup_vs_sort",
        aggfunc="mean",
    )
    table = table.reindex(columns=SPEEDUP_METHOD_ORDER)
    table.columns = [METHOD_LABEL[c] for c in table.columns]
    print(table.round(3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot filtered top-k benchmark results without error bars."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Input CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output PDF path. Default: {DEFAULT_OUT}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_and_filter(args.csv)
    best = best_latency_by_config(df)
    plot_combined(best, args.out)
    print_summary(best)


if __name__ == "__main__":
    main()
