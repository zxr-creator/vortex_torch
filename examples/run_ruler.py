import argparse
import json
import os
import sys
import sglang as sgl
from transformers import AutoTokenizer

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--vortex-module-name", default="block_sparse_attention",
                        help='Use "full_attention" to disable sparsity (dense baseline).')
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3",
                        help='"auto" for bf16/fp16, "fp8_e4m3" for fp8 KV cache.')
    parser.add_argument("--topk-val", type=int, default=29)
    parser.add_argument("--topk-ratio", type=float, default=0.0625)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--mem", type=float, default=0.8)
    parser.add_argument("--tag", default="run",
                        help="Identifier for this run (used in output filename and final print).")
    parser.add_argument("--output-file", default=None,
                        help="Per-sample generations jsonl (default: examples/ruler_output_<tag>.jsonl).")
    args = parser.parse_args()

    model_name = args.model_name
    enable_sparsity = (args.vortex_module_name != "full_attention")

    default_policy = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""

    llm = sgl.Engine(model_path=model_name,
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_block_size=16,
                    vortex_topk_val=args.topk_val,
                    vortex_topk_ratio=args.topk_ratio,
                    disable_overlap_schedule=True,
                    kv_cache_dtype=args.kv_cache_dtype,
                    vortex_dtype="bfloat16",
                    attention_backend="flashinfer",
                    vortex_schedule_policy=default_policy,
                    enable_vortex_sparsity=enable_sparsity,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name=args.vortex_module_name,
                    vortex_max_seq_lens=args.max_new_tokens,
                    mem_fraction_static=args.mem,
                    vortex_workload_chunk_size=32,
                    vortex_compilation_cache_dir="~/.vortex_compilation_cache",
                    )
    
    validation_path = os.path.join(EXAMPLES_DIR, "validation.jsonl")
    with open(validation_path, "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]

    with open(validation_path, "r", encoding="utf-8") as f:
        ruler_outputs = [json.loads(line)["outputs"][0] for line in f]
    
    texts = [
        [{"role":"user","content": x}] for x in ruler_data
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": args.max_new_tokens}
    accuracy = 0
    with open(os.path.join(EXAMPLES_DIR, "ruler_output.jsonl"), "w", encoding="utf-8") as f:
            o = llm.generate(prompts, sampling_params)
            for res, answer in zip(o, ruler_outputs):
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
                    if answer in res["text"]:
                        accuracy += 1.0
    print(f"Ruler Accuracy: {accuracy / len(ruler_outputs) * 100:.2f}%")

if __name__ == "__main__":
    main()
