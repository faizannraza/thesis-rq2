# scripts/eval_with_router.py
import os, orjson, argparse, time, json
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from star_router import STARRouter
from utils_metrics import exact_match, normalized_em, token_f1

def load_jsonl(p):
    rows = []
    with open(p, "rb") as f:
        for line in f: rows.append(orjson.loads(line))
    return rows

def greedy_ensemble_decode(base, tok, adapters: List[Tuple[str,float]], question, device, max_new_tokens=32):
    """
    adapters: list of (adapter_dir, weight)
    At each step, sum logits from each adapter weighted by 'weight'.
    """
    # prepare Peft models once (cache)
    models = []
    for adir, w in adapters:
        peft = PeftModel.from_pretrained(base, adir, is_trainable=False).to(device).eval()
        models.append((peft, w))

    prompt = f"Question: {question}\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    for _ in range(max_new_tokens):
        # get logits from all adapters
        with torch.no_grad():
            logits_sum = None
            for m, w in models:
                out = m(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits[:, -1, :]  # [B, V]
                logits_sum = logits * w if logits_sum is None else logits_sum + w * logits
            next_id = torch.argmax(logits_sum, dim=-1, keepdim=True)  # greedy
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(next_id)], dim=-1)
        # early stop on newline or eos
        if int(next_id[0,0]) == tok.eos_token_id: break
        if tok.decode(next_id[0]).strip() == "": break

    text = tok.decode(input_ids[0], skip_special_tokens=True)
    ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
    return ans

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    dt = torch.float16 if (args.fp16 and device.type=="cuda") else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dt, device_map=None).to(device).eval()

    router = STARRouter(k=args.top_k, encoder=args.encoder, device=str(device))
    router.load(args.router_dir)

    qa = load_jsonl(args.qa_file)[:args.max_eval]
    preds = []
    t0 = time.time()
    for ex in qa:
        weights = router.predict_weights([ex["question"]])[0]  # [(adapter_name, w), ...]
        # map adapter names to dirs under args.adapter_bank_root
        adapters = [(str(Path(args.adapter_bank_root) / name), w) for name, w in weights]
        pred = greedy_ensemble_decode(base, tok, adapters, ex["question"], device, args.max_new_tokens)
        preds.append({"q": ex["question"], "gold": ex["answer"], "pred": pred, "adapters": adapters})

    dur = time.time() - t0
    # metrics
    em = sum(exact_match(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))
    nem = sum(normalized_em(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))
    f1 = sum(token_f1(p["pred"], p["gold"]) for p in preds) / max(1,len(preds))
    out = {"n_eval": len(preds), "EM": em, "nEM": nem, "F1": f1, "seconds": dur,
           "base_model": args.base_model, "router": args.router_dir,
           "qa_file": args.qa_file, "top_k": args.top_k}

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir)/"eval.json", "wb") as f: f.write(orjson.dumps(out))
    with open(Path(args.out_dir)/"preds.jsonl", "wb") as f:
        for r in preds: f.write(orjson.dumps(r)); f.write(b"\n")
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--adapter_bank_root", required=True, help="folder containing per-day adapters (names must match router.meta adapter_names)")
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    main(args)