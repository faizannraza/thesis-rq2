# scripts/eval_with_adapter.py
import os, orjson, argparse, time, pickle
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
    """Generate an answer with greedy decoding."""
    prompt = f"Question: {question}\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
    return ans

def get_embedding(model, tok, text, device):
    """Extract a mean-pooled last hidden state embedding."""
    ids = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.base_model(**ids, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
    emb = hidden.mean(dim=0)               # mean pool tokens → (hidden_dim,)
    return emb

def main(args):
    # Step 1: Device + dtype selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "h100" in gpu_name:
            dt = torch.bfloat16
            print("[Checkpoint] Using CUDA device (H100) with bfloat16 precision")
        else:
            dt = torch.float16
            print(f"[Checkpoint] Using CUDA device ({gpu_name}) with float16 precision")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dt = torch.float32
        print("[Checkpoint] Using Apple MPS backend with float32 precision")
    else:
        device = torch.device("cpu")
        dt = torch.float32
        print("[Checkpoint] Using CPU with float32 precision")

    # Step 2: Tokenizer
    print("[Checkpoint] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Step 3: Base model
    print("[Checkpoint] Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dt,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Step 4: Adapter
    print(f"[Checkpoint] Loading adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    model.to(device).eval()

    # Step 5: Load eval set
    print(f"[Checkpoint] Loading QA file from {args.qa_file}...")
    qa = load_jsonl(args.qa_file)

    # Step 6: Evaluation loop
    total = min(len(qa), args.max_eval)
    print(f"[Checkpoint] Starting evaluation on {total} examples...")
    start = time.time()
    correct = 0
    embeddings = []  # list of {question, embedding}

    for idx, ex in enumerate(qa[:args.max_eval], 1):
        q_text = ex["question"]

        # Prediction
        pred = greedy_answer(model, tok, q_text, device, max_new_tokens=args.max_new_tokens)
        if pred.strip() == ex["answer"].strip():
            correct += 1

        # Embedding (always save as float32 for compatibility)
        emb = get_embedding(model, tok, q_text, device)
        embeddings.append({"question": q_text, "embedding": emb.to(torch.float32).cpu().numpy()})

        if idx % 50 == 0:
            print(f"    Processed {idx}/{total} examples...")

    dur = time.time() - start
    acc = correct / max(1, total)

    # Step 7: Save results + embeddings
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Save metrics
    out = {
        "n_eval": total,
        "exact_match": acc,
        "seconds": dur,
        "base_model": args.base_model,
        "adapter": args.adapter_dir,
        "qa_file": args.qa_file,
    }
    out_path = Path(args.out_dir) / "eval.json"
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(out))

    # Save embeddings
    emb_path = Path(args.out_dir) / "eval_embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)

    print("[Checkpoint] Evaluation complete.")
    print(f"[Result] Exact match: {acc:.3f}, N={total}, Duration={dur:.2f}s")
    print(f"[Result] Results saved to {out_path}")
    print(f"[Result] Embeddings saved to {emb_path}")

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
