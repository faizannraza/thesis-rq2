# scripts/eval_qa.py
import os, orjson, argparse, time, re, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_jsonl(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            rows.append(orjson.loads(line))
    return rows

# --- normalization helpers ---
PUNC_EDGES = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")
ALNUM = re.compile(r"[A-Za-z0-9\-]+")

def norm(s: str) -> str:
    s = s.strip().lower()
    s = PUNC_EDGES.sub("", s)
    return s

def first_alnum_token(s: str) -> str:
    m = ALNUM.search(s)
    return norm(m.group(0)) if m else norm(s)

def build_inputs_answer_tag(tok, question, device):
    # Single-turn, non-chat, with explicit answer tag
    prompt = (
        "Cloze task. Output ONLY the missing token (single word or 4-digit year). "
        "No extra words, no punctuation.\n"
        f"{question}\n"
        "[[ANSWER]]:"
    )
    enc = tok(prompt, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    return enc.to(device), input_len

def decode_continuation(tok, full_ids, input_len):
    gen_ids = full_ids[0, input_len:]            # continuation only
    text = tok.decode(gen_ids, skip_special_tokens=True)
    text = text.split("\n")[0]                   # stop at first newline
    return text

def main(args):
    random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float32

    print(f"Loading model {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dt, device_map=None)
    model.to(device).eval()

    qa = load_jsonl(args.qa_file)
    n = min(len(qa), args.max_eval)
    correct = 0
    debug = []

    t0 = time.time()
    for ex in qa[:n]:
        question = ex["question"]
        gold_raw = ex["answer"]
        gold = norm(gold_raw)

        ids, input_len = build_inputs_answer_tag(tok, question, device)
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            eos_token_id=tok.eos_token_id,
        )

        gen_text = decode_continuation(tok, out, input_len)

        # Heuristic extractor: prefer 4-digit years or Capitalized tokens, else first alnum
        tokens = re.findall(r"[A-Za-z0-9\-]+", gen_text)
        cand = None
        for t in tokens[:5]:                      # inspect first few tokens only
            if re.fullmatch(r"\d{4}", t):
                cand = t; break
        if cand is None:
            for t in tokens[:5]:
                if re.fullmatch(r"[A-Z][a-zA-Z\-]+", t):
                    cand = t; break
        if cand is None and tokens:
            cand = tokens[0]
        pred = norm(cand or "")

        hit = (pred == gold)
        correct += 1 if hit else 0

        if len(debug) < args.debug_examples:
            debug.append({
                "question": question,
                "gold": gold_raw, "gold_norm": gold,
                "gen_text": gen_text, "pred_norm": pred, "hit": hit
            })

    dur = time.time() - t0
    acc = correct / max(1, n)
    out = {
        "n_eval": n, "exact_match": acc,
        "seconds": dur, "sec_per_example": dur/max(1,n),
        "model": args.model_name, "qa_file": args.qa_file
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "eval.json", "wb") as f:
        f.write(orjson.dumps(out))
    print(out)

    if debug:
        with open(Path(args.out_dir) / "debug_samples.jsonl", "wb") as f:
            for row in debug:
                f.write(orjson.dumps(row)); f.write(b"\n")
        print(f"Wrote {len(debug)} debug examples to {Path(args.out_dir) / 'debug_samples.jsonl'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--qa_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_eval", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--debug_examples", type=int, default=10)
    args = ap.parse_args()
    main(args)
