import numpy as np, json, os
from collections import Counter
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/fix_pilot/", exist_ok=True)

OUT = os.path.join(BASE, "out/fix_pilot")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
yT = np.asarray(z2["labels"], dtype=np.int64)[m]
cnt = Counter(yT.tolist())
top20 = [c for c, _ in cnt.most_common(20)]
print("pilot classes:", [classes_sorted[c] for c in top20], flush=True)

zfix = np.load(OUT + "/fix_pilot_plain.npz", allow_pickle=True)
Xfix = norm(zfix["vectors"]); yfix = np.array([name2idx[str(t)] for t in zfix["labels"]], dtype=np.int64)
zn = np.load(BASE + "/data/D_naive_full.npz", allow_pickle=True)
Xnai = norm(zn["vectors"]); ynai = np.array([name2idx[str(t)] for t in zn["labels"]], dtype=np.int64)
zold = np.load(BASE + "/data/D_bedr_train.npz", allow_pickle=True)
Xold = norm(zold["vectors"]); yold = np.array([name2idx[str(t)] for t in zold["labels"]], dtype=np.int64)
XT = norm(z2["vectors"][m])

emb = np.load(BASE + "/data/audit/embeds.npz")
defn = {}
for l, v in zip(emb["def_labels"], emb["def_emb"]):
    defn[name2idx[str(l)]] = v

tmask = np.isin(yT, top20)
Xte, yte = XT[tmask], yT[tmask]
print("eval test rows:", Xte.shape[0], flush=True)

def sample15(X, y, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for c in top20:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=15, replace=False))
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

class CosNC20:
    def fit(self, X, y):
        self.classes_ = top20
        C = np.stack([X[y == c].mean(axis=0) for c in top20])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.W_ = C.astype(np.float32)
        return self
    def predict(self, X):
        idx = (X.astype(np.float32) @ self.W_.T).argmax(axis=1)
        return np.array([self.classes_[i] for i in idx])

def local_entropies(X, y, k=8):
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 1.0)
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    hs = []
    for r in range(len(X)):
        labs = y[part[r]]
        vals, counts = np.unique(labs, return_counts=True)
        p = counts / counts.sum()
        hs.append(float(-(p * np.log2(p)).sum()))
    return np.array(hs)

def stats(X, y):
    rec = {}
    for c in top20:
        Xc = X[y == c]
        S = Xc @ Xc.T
        ii, jj = np.triu_indices(len(Xc), k=1)
        cent = Xc.mean(axis=0)
        cn = cent / (np.linalg.norm(cent) + 1e-12)
        mm = yte == c
        rec[c] = {
            "mean_pair": float(S[ii, jj].mean()),
            "max_pair": float(S[ii, jj].max()),
            "test_cov": float((Xte[mm] @ cn).mean()),
            "fidelity": float((Xc @ (defn[c] / (np.linalg.norm(defn[c]) + 1e-12))).mean()),
        }
    return rec

rows = {}
for name, (X, y) in [("FIX", (Xfix, yfix)), ("NAIVE", (Xnai, ynai)), ("OLD", (Xold, yold))]:
    acc_nc, acc_ridge = [], []
    for seed in [42, 43, 44]:
        Xs, ys = sample15(X, y, seed)
        acc_nc.append(accuracy_score(yte, CosNC20().fit(Xs, ys).predict(Xte)))
        acc_ridge.append(accuracy_score(yte, RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys).predict(Xte)))
    Xs42, ys42 = sample15(X, y, 42)
    st = stats(Xs42, ys42)
    ent = local_entropies(Xs42, ys42)
    rows[name] = {
        "NC_acc": [round(float(np.mean(acc_nc)), 4), round(float(np.std(acc_nc)), 4)],
        "Ridge_acc": [round(float(np.mean(acc_ridge)), 4), round(float(np.std(acc_ridge)), 4)],
        "fidelity": round(float(np.mean([v["fidelity"] for v in st.values()])), 4),
        "mean_pair": round(float(np.mean([v["mean_pair"] for v in st.values()])), 4),
        "max_pair": round(float(np.mean([v["max_pair"] for v in st.values()])), 4),
        "test_cov": round(float(np.mean([v["test_cov"] for v in st.values()])), 4),
        "selfpool_entropy": round(float(ent.mean()), 4),
    }
    print(name, json.dumps(rows[name]), flush=True)
json.dump(rows, open(OUT + "/results.json", "w"), indent=1)
print("DONE", flush=True)

