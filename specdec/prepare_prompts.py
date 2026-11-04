"""One-time CPU step: freeze WikiText-2 test prompts to results/prompts_wikitext2.json.

Prompt = first `prompt_len` GPT-2 tokens of a shuffled, non-heading paragraph that
has at least 2x prompt_len tokens. Usage: python -m specdec.prepare_prompts
"""
import argparse
import json
import random

from . import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # read the parquet shard directly: the `datasets` library would import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    texts = pq.read_table(path).column("text").to_pylist()
    paras = [t.strip() for t in texts if t.strip() and not t.strip().startswith("=")]
    random.Random(args.seed).shuffle(paras)
    tok = common.tokenizer()
    out = []
    for p in paras:
        ids = tok.encode(p).ids
        if len(ids) >= 2 * args.prompt_len:
            out.append(ids[:args.prompt_len])
        if len(out) == args.n:
            break
    with open(common.PROMPTS, "w") as f:
        json.dump({"source": "Salesforce/wikitext wikitext-2-raw-v1 test", "seed": args.seed,
                   "prompt_len": args.prompt_len, "prompts": out}, f)
    print(f"wrote {len(out)} prompts to {common.PROMPTS}")
    print("example:", tok.decode(out[0]))
    common.assert_no_torch()


if __name__ == "__main__":
    main()
