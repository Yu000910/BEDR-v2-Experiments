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

def load_dataset(npz_path, name_kind):
    z = np.load(npz_path, allow_pickle=True)
    X = norm(z["vectors"])
    y = np.array([name2idx[str(t)] for t in z["labels"]], dtype=np.int64)
    return X, y

datasets = {
    "FIX2": load_dataset(OUT + "/fix2_pilot_plain.npz", "name"),
    "NAIVE": load_dataset(BASE + "/data/D_naive_full.npz", "name"),
    "OLD": load_dataset(BASE + "/data/D_bedr_train.npz", "name"),
}
XT = norm(z2["vectors"][m])
emb = np.load(BASE + "/data/audit/embeds.npz")
defn = {}
for l, v in zip(emb["def_labels"], emb["def_emb"]):
    defn[name2idx[str(l)]] = v

tmask = np.isin(yT, top20)
Xte, yte = XT[tmask], yT[tmask]
print("eval test rows:", Xte.shape[0], flush=True)
for name, (X, y) in datasets.items():
    print(name, "rows", X.shape[0], "per-class:", sorted(Counter(y.tolist()).values())[:3], "...", flush=True)

def samplek(X, y, k, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for c in top20:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False) if len(ci) >= k else ci)
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

results = {}
for k in [15, 30]:
    for name, (X, y) in datasets.items():
        acc_nc, acc_ridge = [], []
        for seed in [42, 43, 44]:
            Xs, ys = samplek(X, y, k, seed)
            acc_nc.append(accuracy_score(yte, CosNC20().fit(Xs, ys).predict(Xte)))
            acc_ridge.append(accuracy_score(yte, RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys).predict(Xte)))
        Xs42, ys42 = samplek(X, y, k, 42)
        st = stats(Xs42, ys42)
        results["k%d|%s" % (k, name)] = {
            "NC_acc": [round(float(np.mean(acc_nc)), 4), round(float(np.std(acc_nc)), 4)],
            "Ridge_acc": [round(float(np.mean(acc_ridge)), 4), round(float(np.std(acc_ridge)), 4)],
            "fidelity": round(float(np.mean([v["fidelity"] for v in st.values()])), 4),
            "mean_pair": round(float(np.mean([v["mean_pair"] for v in st.values()])), 4),
            "max_pair": round(float(np.mean([v["max_pair"] for v in st.values()])), 4),
            "test_cov": round(float(np.mean([v["test_cov"] for v in st.values()])), 4),
        }
        print("k%d|%s" % (k, name), json.dumps(results["k%d|%s" % (k, name)]), flush=True)
json.dump(results, open(OUT + "/results_fix2.json", "w"), indent=1)
print("DONE", flush=True)

