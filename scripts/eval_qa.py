# scripts/eval_qa.py
import os, orjson, argparse, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_jsonl(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            rows.append(orjson.loads(line))
    return rows

def pick_device_dtype(args):
    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device in ("cpu", "auto"):
        # auto -> prefer cuda, then mps, else cpu
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # fallback
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    # dtype
    if device.type == "cuda":
        if args.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
        elif args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    return device, dtype

def greedy_answer(model, tok, question, device, max_new_tokens=32):
    prompt = f"Question: {question}\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
    return ans

def main(args):
    device, dtype = pick_device_dtype(args)
    print(f"Loading model {args.model_name} on {device} dtype={dtype} ...")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # eval path doesn't need 4-bit; keep it simple & fast
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    qa = load_jsonl(args.qa_file)
    start = time.time()
    correct = 0
    for ex in qa[:args.max_eval]:
        pred = greedy_answer(model, tok, ex["question"], device, max_new_tokens=args.max_new_tokens)
        if pred.strip() == ex["answer"].strip():
            correct += 1
    dur = time.time() - start
    n = min(len(qa), args.max_eval)
    acc = correct / max(1, n)

    out = {
        "n_eval": n, "exact_match": acc, "seconds": dur,
        "sec_per_example": (dur / max(1, n)),
        "model": args.model_name, "qa_file": args.qa_file
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "eval.json", "wb") as f:
        f.write(orjson.dumps(out))
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", choices=["auto","cuda","mps","cpu"], default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    main(args)
