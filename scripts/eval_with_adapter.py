# scripts/eval_with_adapter.py
import os, orjson, argparse, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_jsonl(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            rows.append(orjson.loads(line))
    return rows

def greedy_answer(model, tok, question, device, max_new_tokens=32):
    prompt = f"Question: {question}\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
    return ans

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dt, device_map=None)
    base.to(device).eval()

    # load adapter
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    model.to(device).eval()

    qa = load_jsonl(args.qa_file)
    start = time.time()
    correct = 0
    for ex in qa[:args.max_eval]:
        pred = greedy_answer(model, tok, ex["question"], device, max_new_tokens=args.max_new_tokens)
        correct += 1 if pred.strip() == ex["answer"].strip() else 0
    dur = time.time() - start
    n = min(len(qa), args.max_eval)
    acc = correct / max(1, n)

    out = {"n_eval": n, "exact_match": acc, "seconds": dur,
           "base_model": args.base_model, "adapter": args.adapter_dir, "qa_file": args.qa_file}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "eval.json", "wb") as f:
        f.write(orjson.dumps(out))
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()
    main(args)
