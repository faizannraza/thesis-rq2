# scripts/prepare_days.py
# Build 30 synthetic "days" from a text corpus and produce cloze-style QA.
# Produces:
#  - data/days/qa_train_day_XX.jsonl
#  - data/days/qa_eval_day_XX.jsonl
#  - data/legacy/qa_legacy_holdout.jsonl

import re, os, orjson, random, argparse, math
from datasets import load_dataset
from pathlib import Path

random.seed(42)

def clean_text(t):
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def sentences_from_text(text):
    # lightweight sentence split (enough for our purpose)
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(s) for s in sents if len(s.split()) >= 8 and len(s.split()) <= 25]

def pick_cloze(sent):
    # pick a target token: prefer ProperNouns / Numbers heuristically
    tokens = sent.split()
    # indices with a Capitalized word not at beginning, or a 4-digit number
    cand_idx = [i for i,w in enumerate(tokens)
                if (re.match(r"^[A-Z][a-zA-Z-]+$", w) and i>0) or re.match(r"^\d{4}$", w)]
    if not cand_idx:
        return None
    idx = random.choice(cand_idx)
    answer = tokens[idx]
    masked = tokens[:]
    masked[idx] = "[[BLANK]]"
    masked_sent = " ".join(masked)
    q = f"Fill the blank in this sentence: '{masked_sent}'"
    return q, answer

def make_qa_from_paragraph(paragraph, max_q=3):
    sents = sentences_from_text(paragraph)
    random.shuffle(sents)
    qa = []
    for s in sents:
        cloze = pick_cloze(s)
        if cloze:
            q, a = cloze
            qa.append({"question": q, "answer": a})
            if len(qa) >= max_q:
                break
    return qa

def save_jsonl(path, rows):
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

def main(args):
    out_days = Path(args.days_dir)
    out_legacy = Path(args.legacy_dir)
    out_days.mkdir(parents=True, exist_ok=True)
    out_legacy.mkdir(parents=True, exist_ok=True)

    # Load wikitext (smallish) to keep RAM modest; you can switch to bigger if you want
    # Options: "wikitext", "wikitext-103-raw-v1" but it's larger
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    # Concatenate train split paragraphs
    all_text = "\n".join(ds["train"]["text"])
    paragraphs = [p for p in all_text.split("\n\n") if len(p.split()) > 40][:20000]  # cap for speed

    # Shuffle and bin into 30 days (balanced by count)
    random.shuffle(paragraphs)
    n_days = args.days
    per_day = math.ceil(len(paragraphs) / n_days)
    day_bins = [paragraphs[i*per_day:(i+1)*per_day] for i in range(n_days)]

    # Build LEGACY pool from the first 10% of the corpus (held-out, never updated)
    legacy_paras = paragraphs[: max(500, len(paragraphs)//10)]
    legacy_qa = []
    for p in legacy_paras[:3000]:  # cap for speed
        legacy_qa += make_qa_from_paragraph(p, max_q=2)
    # Deduplicate Qs
    seen = set()
    legacy_qa_dedup = []
    for ex in legacy_qa:
        k = (ex["question"], ex["answer"])
        if k not in seen:
            legacy_qa_dedup.append(ex)
            seen.add(k)
    save_jsonl(out_legacy / "qa_legacy_holdout.jsonl", legacy_qa_dedup[:5000])

    # For each day, make train/eval QA from that day's paras
    for d, paras in enumerate(day_bins, start=1):
        train, eval_ = [], []
        for i, p in enumerate(paras[:3000]):  # cap day size reasonably for M1
            qa = make_qa_from_paragraph(p, max_q=2)
            # split ~70/30 into train/eval
            for j, ex in enumerate(qa):
                (train if j % 3 != 0 else eval_).append(ex)

        save_jsonl(out_days / f"qa_train_day_{d:02d}.jsonl", train)
        save_jsonl(out_days / f"qa_eval_day_{d:02d}.jsonl", eval_)

    print("Prepared:")
    print(f" - Legacy eval: {out_legacy / 'qa_legacy_holdout.jsonl'}")
    print(f" - 30 days in: {out_days}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days_dir", default="data/days")
    ap.add_argument("--legacy_dir", default="data/legacy")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    main(args)
