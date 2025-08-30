# scripts/train_lora.py
# Train a LoRA adapter on a JSONL QA file (question -> answer)
import os, argparse, orjson, math, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

random.seed(42)
torch.manual_seed(42)

class QADataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.rows = []
        with open(path, "rb") as f:
            for line in f:
                self.rows.append(orjson.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt = f"Question: {ex['question']}\nAnswer:"
        target = ex["answer"]
        text = prompt + " " + target
        ids = self.tok(text, truncation=True, max_length=self.max_len)
        return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]}

def train_lora(
    base_model, train_jsonl, out_dir,
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    epochs=3, bsz=1, grad_accum=8, lr=2e-4, max_len=512
):
    # Step 1: Device + dtype
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
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Step 3: Base model
    print("[Checkpoint] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dt,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        print("[Checkpoint] Enabled gradient checkpointing for memory efficiency")

    # Step 4: LoRA adapter
    print("[Checkpoint] Attaching LoRA adapter...")
    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Step 5: Dataset
    print(f"[Checkpoint] Loading training data from {train_jsonl}...")
    ds_train = QADataset(train_jsonl, tok, max_len=max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Step 6: Trainer setup
    print("[Checkpoint] Setting up Trainer...")
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=25,
        save_strategy="no",
        fp16=(dt == torch.float16),
        bf16=(dt == torch.bfloat16),
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        train_dataset=ds_train,
        args=args,
        data_collator=collator,
    )

    # Step 7: Training
    print(f"[Checkpoint] Starting training for {epochs} epoch(s)...")
    start = time.time()
    trainer.train()
    dur = time.time() - start
    print(f"[Checkpoint] Training finished in {dur:.2f} seconds")

    # Step 8: Save adapter
    print(f"[Checkpoint] Saving adapter + tokenizer to {out_dir}...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    return {"train_seconds": dur, "n_train": len(ds_train)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    stats = train_lora(
        base_model=args.base_model,
        train_jsonl=args.train_jsonl,
        out_dir=args.out_dir,
        epochs=args.epochs,
        bsz=args.bsz,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_len=args.max_len,
    )
    print("[Result]", stats)
