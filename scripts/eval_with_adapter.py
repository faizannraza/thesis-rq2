# scripts/eval_with_adapter.py  (drop-in replacement)
import os, orjson, argparse, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils_metrics import exact_match, normalized_em, token_f1

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
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = torch.float16 if (args.fp16 and device.type=="cuda") else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dt, device_map=None).to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False).to(device).eval()

    qa = load_jsonl(args.qa_file)[:args.max_eval]
    preds = []
    start = time.time()
    for ex in qa:
        pred = greedy_answer(model, tok, ex["question"], device, max_new_tokens=args.max_new_tokens)
        preds.append({"q": ex["question"], "gold": ex["answer"], "pred": pred})
    dur = time.time() - start

    em  = sum(exact_match(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))
    nem = sum(normalized_em(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))
    f1  = sum(token_f1(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))

    out = {"n_eval": len(preds), "exact_match": em, "normalized_em": nem, "f1": f1,
           "seconds": dur, "base_model": args.base_model, "adapter": args.adapter_dir,
           "qa_file": args.qa_file}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "eval.json", "wb") as f: f.write(orjson.dumps(out))
    with open(Path(args.out_dir) / "preds.jsonl", "wb") as f:
        for r in preds: f.write(orjson.dumps(r)); f.write(b"\n")
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    main(args)
