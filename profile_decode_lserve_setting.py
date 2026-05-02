#!/usr/bin/env python3
"""Vortex profile_decode adapted to match the OmniServe LServe test setting.

Mirrors the parameters exercised in tests/lserve_kv_smoke.py on the OmniServe
side so the same prompt size, batch shape, and decode budget are benchmarked
under vortex_torch. Compares two KV-width regimes:

  - Vortex fp8 vs OmniServe kv8 (--kv-cache-dtype fp8_e5m2  ↔  --precision w8a8kv8)
  - Vortex bf16 vs OmniServe kv16 (--kv-cache-dtype auto    ↔  --precision w16a16kv16)

Side-by-side, the two should be apples-to-apples on:
  - model:        meta-llama/Meta-Llama-3-8B-Instruct (fp16/bf16 weights)
  - batch size:   1
  - prefill len:  8192 (random-tokenized; matches lserve_kv_smoke.py default)
  - decode steps: 64
  - sparsity:     vortex BLOCK_TOPK with --vortex-topk-val mapped to roughly the
                  same effective KV-page count as LServe's
                  dynamic_sparse_token_budget=4096

Differences that are NOT squashable here:
  - Sparsity granularity: LServe selects KV pages of 64 tokens × dynamic budget
    4096 via min/max pooling; vortex's BLOCK_TOPK selects K pages of `page_size`
    tokens. --vortex-topk-val=8 + --page-size=512 → ~4096 tokens of KV ≈ same
    budget. Tweak --vortex-topk-val if your benchmark needs a different budget.
  - 8-bit format: omniserve uses INT8 per-tensor scaled; vortex uses fp8_e5m2
    (or fp8_e4m3). These are different numerical formats but both 1 byte/elem.

Usage (from /root/vortex_torch, with the vortex venv):
  # 8-bit KV (vortex fp8 vs omniserve int8):
  /root/vortex/bin/python profile_decode_lserve_setting.py --kv-bits 8

  # 16-bit KV (vortex bf16 vs omniserve fp16):
  /root/vortex/bin/python profile_decode_lserve_setting.py --kv-bits 16
"""
import argparse
import dataclasses
import logging
import random
from pathlib import Path
from typing import List, Optional

import torch

from sglang.bench_one_batch import decode, extend, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger


def _build_reqs(batch_size, tokenizer, max_new_tokens, input_len, random_seed=0):
    sampling = SamplingParams(temperature=0, max_new_tokens=max_new_tokens)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    rng = random.Random(random_seed)
    input_ids_per_req = [
        [rng.randrange(vocab_size) for _ in range(input_len)]
        for _ in range(batch_size)
    ]
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids_per_req[i]),
            sampling_params=sampling,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
        print(f"Request {i} length (tokens): {len(input_ids_per_req[i])}")
    return reqs


def main():
    parser = argparse.ArgumentParser(
        description="vortex_torch decode benchmark mirroring OmniServe LServe kv8 setting"
    )
    ServerArgs.add_cli_args(parser)

    # OmniServe-mirroring defaults
    parser.add_argument("--kv-bits", type=int, choices=[8, 16], default=8,
                        help="8 → fp8_e5m2 (vs omniserve int8); "
                             "16 → auto/bf16 (vs omniserve fp16)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size; matches lserve_kv_smoke (1)")
    parser.add_argument("--input-len", type=int, default=8192,
                        help="prefill length in tokens; matches lserve_kv_smoke (8192)")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="decode steps to time; matches typical LServe smoke")
    parser.add_argument("--vortex-algorithm", default="BLOCK_TOPK")
    parser.add_argument("--vortex-topk-val", type=int, default=8,
                        help="top-K KV pages per decode step. With --page-size=512, "
                             "8*512=4096 tokens ≈ LServe dynamic_sparse_token_budget=4096")

    args = parser.parse_args()

    # Backfill ServerArgs defaults that aren't exposed via add_cli_args
    for f in dataclasses.fields(ServerArgs):
        if hasattr(args, f.name):
            continue
        if f.default is not dataclasses.MISSING:
            setattr(args, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore
            setattr(args, f.name, f.default_factory())  # type: ignore
        else:
            setattr(args, f.name, None)
    server_args = ServerArgs.from_cli_args(args)

    # Lock vortex_torch sparse-attention configuration
    server_args.attention_backend = "flashinfer"
    server_args.enable_vortex_sparsity = True
    server_args.disable_overlap_schedule = True
    server_args.disable_cuda_graph = False
    server_args.vortex_module_name = "block_sparse_attention"
    server_args.vortex_topk_val = args.vortex_topk_val
    server_args.vortex_layers_skip = [0]
    server_args.page_size = 512
    server_args.block_size = 16
    server_args.vortex_block_reserved_bos = 1
    server_args.vortex_block_reserved_eos = 2
    server_args.vortex_workload_chunk_size = 32
    server_args.vortex_compilation_cache_dir = "./vortex_compilation_cache"
    server_args.vortex_max_seq_lens = max(args.input_len + args.max_new_tokens, 16384)
    server_args.mem_fraction_static = 0.85

    # Model: same Llama-3-8B-Instruct used in the OmniServe smoke.
    # Default to the local fp16 mirror downloaded for the kv16 leg if present.
    if server_args.model_path is None:
        local_path = "/root/omniserve/models/llama-3-8b-Instruct-fp16"
        server_args.model_path = local_path if Path(local_path).is_dir() else "unsloth/llama-3-8b-Instruct"

    # KV cache dtype: 8-bit → fp8_e5m2; 16-bit → keep "auto" so SGLang uses model dtype.
    # If the user explicitly set --kv-cache-dtype on the CLI, leave it alone.
    if server_args.kv_cache_dtype == "auto" and args.kv_bits == 8:
        server_args.kv_cache_dtype = "fp8_e5m2"
    # 16-bit case: leave kv_cache_dtype="auto" → SGLang stores KV at the model dtype.

    if server_args.tp_size != 1:
        raise ValueError("This script is single-GPU only. Use tp_size=1.")

    print(
        f"[setting] model={server_args.model_path} batch={args.batch_size} "
        f"input_len={args.input_len} max_new_tokens={args.max_new_tokens} "
        f"kv_cache_dtype={server_args.kv_cache_dtype} "
        f"vortex_topk={server_args.vortex_topk_val} page_size={server_args.page_size}"
    )

    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()),
                        format="%(message)s")
    _set_envs_and_config(server_args)
    configure_logger(server_args, prefix=" TP0")

    port_args = PortArgs.init_new(server_args)
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank=0)

    # Resolved backend
    backend_obj = getattr(model_runner, "attn_backend", None)
    backend_cls = backend_obj.__class__ if backend_obj is not None else None
    print(
        "Resolved attention backend:",
        f"{backend_cls.__module__}.{backend_cls.__name__}" if backend_cls else "<none>",
    )

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    reqs = _build_reqs(args.batch_size, tokenizer, args.max_new_tokens, args.input_len)

    with torch.no_grad():
        per_req_next = []
        batch = None
        for i, req in enumerate(reqs):
            with torch.cuda.nvtx.range(f"prefill_req_{i}"):
                nti, _, batch_i = extend([req], model_runner)
            per_req_next.append(nti)
            batch = batch_i if batch is None else (batch.merge_batch(batch_i) or batch)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        next_token_ids = torch.cat(per_req_next, dim=0)
        torch.cuda.synchronize()

        # Time decode
        decode_steps = max(args.max_new_tokens - 1, 0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.nvtx.range("decode_loop"):
            for step in range(decode_steps):
                next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        end.record()
        torch.cuda.synchronize()
        dt_ms = start.elapsed_time(end)

    total_decoded = decode_steps * args.batch_size
    print()
    print("=" * 60)
    print(f"[result] vortex_torch ({server_args.kv_cache_dtype}) sparse=BLOCK_TOPK k={server_args.vortex_topk_val}*{server_args.page_size}")
    print(f"[result] decode_steps={decode_steps} batch={args.batch_size} -> {total_decoded} tokens")
    print(f"[result] decode_total_ms={dt_ms:.2f}")
    print(f"[result] decode_ms_per_token={dt_ms / max(total_decoded,1):.4f}")
    print(f"[result] decode_tokens_per_sec={total_decoded * 1000.0 / max(dt_ms,1e-9):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
