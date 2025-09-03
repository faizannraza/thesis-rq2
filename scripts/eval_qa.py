# scripts/eval_qa.py
import os, orjson, argparse, time, re
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
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

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

def normalize(s: str) -> str:
    """Lowercase, strip quotes/punct, collapse whitespace."""
    s = s.strip()
    s = s.strip("‘’“”'\"`.,;:!?()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def build_prompt(question: str, tok, use_chat_template: bool, system_prompt: str | None):
    """
    Build a strict 'single-token only' prompt. If chat template is requested
    and available on the tokenizer, apply it; otherwise use a plain prompt.
    """
    default_system = (
        "You are a strict QA assistant.\n"
        "Task: Fill the blank. Output ONLY the single missing token — no quotes, "
        "no punctuation, no extra words."
    )
    sys_text = system_prompt if system_prompt else default_system

    if use_chat_template and hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": f"{question}\nAnswer:"},
        ]
        # add_generation_prompt ensures model starts generating assistant turn
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    else:
        # plain text fallback
        return f"{sys_text}\n\nQuestion: {question}\nAnswer:"

def greedy_answer(model, tok, prompt: str, device, max_new_tokens=8):
    """
    Generate with tight constraints. We still decode the first line and
    keep only the first whitespace-delimited token.
    """
    ids = tok(prompt, return_tensors="pt").to(device)
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Try to isolate what's after "Answer:" if present; otherwise take last line
    if "Answer:" in text:
        ans = text.split("Answer:", 1)[-1].strip()
    else:
        # fallback: take the tail after the prompt tokens
        ans = text[len(prompt):].strip()
    ans = ans.splitlines()[0].strip()
    # keep only first token (cloze targets are single-token by design)
    ans = ans.split()[0] if ans else ""
    ans = normalize(ans)
    return ans

def main(args):
    device, dtype = pick_device_dtype(args)
    print(f"Loading model {args.model_name} on {device} dtype={dtype} ...")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
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
    debug_rows = []

    for ex in qa[:args.max_eval]:
        prompt = build_prompt(
            question=ex["question"],
            tok=tok,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt
        )
        pred = greedy_answer(
            model, tok, prompt, device,
            max_new_tokens=args.max_new_tokens
        )
        gold = normalize(ex["answer"])
        ok = (pred == gold)
        correct += 1 if ok else 0

        if args.debug_examples and len(debug_rows) < args.debug_examples:
            debug_rows.append({
                "question": ex["question"],
                "gold": gold,
                "pred": pred,
                "match": ok
            })

    dur = time.time() - start
    n = min(len(qa), args.max_eval)
    acc = correct / max(1, n)

    out = {
        "n_eval": n, "exact_match": acc, "seconds": dur,
        "sec_per_example": (dur / max(1, n)),
        "model": args.model_name, "qa_file": args.qa_file,
        "chat_template": bool(args.use_chat_template)
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "eval.json", "wb") as f:
        f.write(orjson.dumps(out))
    print(out)

    if debug_rows:
        dbg_path = Path(args.out_dir) / "debug_samples.jsonl"
        with open(dbg_path, "wb") as f:
            for r in debug_rows:
                f.write(orjson.dumps(r)); f.write(b"\n")
        print(f"Wrote {len(debug_rows)} debug examples to {dbg_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", choices=["auto","cuda","mps","cpu"], default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Use tokenizer.apply_chat_template if available (e.g., GPT-OSS harmony).")
    ap.add_argument("--system_prompt", type=str, default=None,
                    help="Override the strict one-token system prompt.")
    ap.add_argument("--debug_examples", type=int, default=0,
                    help="Write N debug rows to debug_samples.jsonl")
    args = ap.parse_args()
    main(args)
