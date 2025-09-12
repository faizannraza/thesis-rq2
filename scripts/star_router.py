# scripts/star_router.py
import os, json, math, random, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Embeddings: prefer sentence-transformers, else TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    SENT_EMB_OK = True
except Exception:
    SENT_EMB_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(42); np.random.seed(42); torch.manual_seed(42)

def topk_softmax(logits, k):
    # keep only top-k, renormalize
    topv, topi = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(1, topi, topv)
    return F.softmax(mask, dim=-1)

class TextFeaturizer:
    def __init__(self, method="auto"):
        self.method = method
        self.model = None
        self.vec = None

    def fit(self, texts: List[str]):
        if self.method == "sbert" or (self.method=="auto" and SENT_EMB_OK):
            self.method = "sbert"
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.method = "tfidf"
            self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            self.vec.fit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.method == "sbert":
            X = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return X.astype(np.float32)
        else:
            X = self.vec.transform(texts)
            X = normalize(X, norm="l2", axis=1)
            return X.astype(np.float32).toarray()

class RouterMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, n_adapters=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_adapters)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)  # logits

class STARRouter:
    """
    Router that maps a query to a sparse distribution over an adapter pool.
    Trained with soft targets derived from which adapters answer eval Qs best.
    """
    def __init__(self, k=2, encoder="auto", device=None):
        self.k = k
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feat = TextFeaturizer(encoder)
        self.mlp = None
        self.adapter_names: List[str] = []

    def fit(self, qs: List[str], soft_targets: np.ndarray, adapter_names: List[str],
            epochs=8, lr=1e-3, batch=128):
        self.adapter_names = adapter_names
        self.feat.fit(qs)
        X = torch.tensor(self.feat.encode(qs), dtype=torch.float32, device=self.device)
        Y = torch.tensor(soft_targets, dtype=torch.float32, device=self.device)
        in_dim = X.shape[1]
        self.mlp = RouterMLP(in_dim, hidden=256, n_adapters=len(adapter_names)).to(self.device)
        opt = torch.optim.Adam(self.mlp.parameters(), lr=lr, weight_decay=1e-5)

        n = X.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                xb, yb = X[idx], Y[idx]
                logits = self.mlp(xb)
                # encourage sparsity by training on top-k softmax
                probs = topk_softmax(logits, self.k)
                loss = F.mse_loss(probs, yb)
                # small L1 regularizer to push logits down
                loss = loss + 1e-4 * torch.mean(torch.abs(logits))
                opt.zero_grad(); loss.backward(); opt.step()

    def predict_weights(self, qs: List[str]) -> List[List[Tuple[str, float]]]:
        X = torch.tensor(self.feat.encode(qs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.mlp(X)
            probs = topk_softmax(logits, self.k)  # [B, A]
            topv, topi = torch.topk(probs, k=self.k, dim=-1)
        out = []
        for rowv, rowi in zip(topv.cpu().tolist(), topi.cpu().tolist()):
            pairs = [(self.adapter_names[j], float(v)) for j, v in zip(rowi, rowv)]
            out.append(pairs)
        return out

    def save(self, path: str):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.mlp.state_dict(), path / "router.pt")
        with open(path / "router.meta.json", "w") as f:
            json.dump({"k": self.k, "adapter_names": self.adapter_names,
                       "encoder": self.feat.method}, f)
        # store TF-IDF model if used
        if self.feat.method == "tfidf":
            import joblib
            joblib.dump(self.feat.vec, path / "tfidf.joblib")

    def load(self, path: str):
        path = Path(path)
        meta = json.loads((path / "router.meta.json").read_text())
        self.k = meta["k"]; self.adapter_names = meta["adapter_names"]
        enc = meta["encoder"]
        self.feat = TextFeaturizer(enc)
        if enc == "tfidf":
            import joblib
            self.feat.vec = joblib.load(path / "tfidf.joblib")
        else:
            self.feat.fit(["dummy"])  # load SBERT
        # init mlp
        in_dim = (self.feat.model.get_sentence_embedding_dimension()
                  if enc=="sbert" else self.feat.vec.max_features)
        self.mlp = RouterMLP(in_dim, hidden=256, n_adapters=len(self.adapter_names)).to(self.device)
        self.mlp.load_state_dict(torch.load(path / "router.pt", map_location=self.device))
        self.mlp.eval()