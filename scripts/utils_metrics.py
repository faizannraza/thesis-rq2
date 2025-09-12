# scripts/utils_metrics.py
import re, math
from collections import Counter

PUNC_EDGES = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")

def norm(s: str) -> str:
    s = s.strip().lower()
    s = PUNC_EDGES.sub("", s)
    return s

def token_f1(pred: str, gold: str) -> float:
    p = norm(pred).split()
    g = norm(gold).split()
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    cp = Counter(p); cg = Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0: return 0.0
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cg.values()))
    return 2 * prec * rec / max(1e-8, (prec + rec))

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if norm(pred) == norm(gold) else 0.0

def normalized_em(pred: str, gold: str) -> float:
    # “lenient” EM: accept first alnum token equality or 4-digit year match
    pred_n = norm(pred); gold_n = norm(gold)
    if pred_n == gold_n: return 1.0
    tok = re.findall(r"[A-Za-z0-9\-]+", pred)
    if not tok: return 0.0
    first = norm(tok[0])
    if first == gold_n: return 1.0
    if re.fullmatch(r"\d{4}", first) and re.fullmatch(r"\d{4}", gold_n):
        return 1.0 if first == gold_n else 0.0
    return 0.0