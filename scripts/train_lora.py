# scripts/train_lora.py
import os, argparse, orjson, math, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, BitsAndBytesConfig
)
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

def pick_device_dtype(args):
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device in ("cpu", "auto"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

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

def train_lora(
    base_model, train_jsonl, out_dir,
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    epochs=3, bsz=1, grad_accum=8, lr=2e-4, max_len=512,
    device="auto", bf16=False, fp16=False, load_4bit=False, gradient_checkpointing=False
):
    device, dtype = pick_device_dtype(argparse.Namespace(device=device, bf16=bf16, fp16=fp16))
    print(f"Training on {device} dtype={dtype} load_4bit={load_4bit}")

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    quant_config = None
    if load_4bit and device.type == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16
        )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model)
    if hasattr(config, "quantization_config") and config.quantization_config is None:
        delattr(config, "quantization_config")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=quant_config
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ds_train = QADataset(train_jsonl, tok, max_len=max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Optimizer choice: 8-bit optimizer if we loaded 4-bit weights
    optim = "adamw_torch"
    if load_4bit and device.type == "cuda":
        try:
            import bitsandbytes as bnb  # noqa: F401
            optim = "paged_adamw_8bit"
        except Exception:
            optim = "adamw_torch"

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=25,
        save_strategy="no",
        fp16=(fp16 and device.type == "cuda"),
        bf16=(bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()),
        optim=optim,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        train_dataset=ds_train,
        args=args,
        data_collator=collator,
    )

    start = time.time()
    trainer.train()
    dur = time.time() - start

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
    ap.add_argument("--device", choices=["auto","cuda","mps","cpu"], default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
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
        device=args.device,
        bf16=args.bf16,
        fp16=args.fp16,
        load_4bit=args.load_4bit,
        gradient_checkpointing=args.gradient_checkpointing
    )
    print(stats)
