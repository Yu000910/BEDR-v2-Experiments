import numpy as np, json, os
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/tail_flicker/", exist_ok=True)

OUT = os.path.join(BASE, "out/tail_flicker")
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(os.path.join(BASE, "data/classes_sorted.json")))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def load_npz(path, kind):
    z = np.load(path, allow_pickle=True)
    X = z["vectors"].astype(np.float32)
    y = np.array([name2idx[str(t)] for t in z["labels"]], dtype=np.int64) if kind == "name" else np.asarray(z["labels"], dtype=np.int64)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n, y

BX, By = load_npz(os.path.join(BASE, "data/D_bedr_train.npz"), "name")
NX, Ny = load_npz(os.path.join(BASE, "data/D_naive_full.npz"), "name")
z2 = np.load(os.path.join(BASE, "data/test2_full_plain.npz"), allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
TX = z2["vectors"][m].astype(np.float32); Ty = np.asarray(z2["labels"], dtype=np.int64)[m]
nn = np.linalg.norm(TX, axis=1, keepdims=True); nn[nn == 0] = 1.0
TX = TX / nn

cb, cn = Counter(By.tolist()), Counter(Ny.tolist())
ctB = set(Ty.tolist())
tail = sorted([c for c in cb if cb[c] < 10 and cn[c] >= 5 and c in ctB])
print("tail classes n=%d:" % len(tail), [classes_sorted[c] for c in tail], flush=True)
mask = np.isin(Ty, tail)
Xte, yte = TX[mask], Ty[mask]
print("tail test rows:", Xte.shape[0], flush=True)

class CosNC:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        C = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.W_ = C.astype(np.float32)
        return self
    def predict(self, X):
        idx = (X.astype(np.float32) @ self.W_.T).argmax(axis=1)
        return self.classes_[idx]

def judge(name, seed):
    if name == "NC": return CosNC()
    if name == "KNN1": return KNeighborsClassifier(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    if name == "KNN5": return KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1)
    if name == "SGD": return SGDClassifier(loss="log_loss", max_iter=30, alpha=1e-4, random_state=seed, n_jobs=-1)
    if name == "RIDGE": return RidgeClassifier(alpha=1.0, random_state=seed)
    raise ValueError(name)

def sub5(X, y, seed, classes):
    rng = np.random.RandomState(seed)
    parts = []
    for c in classes:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=5, replace=False))
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

SEEDS = list(range(42, 52))
results = {}
for jname in ["NC", "KNN1", "KNN5", "SGD", "RIDGE"]:
    bedr, na, nb = [], [], []
    for seed in SEEDS:
        Xb, yb = sub5(BX, By, seed, tail)
        Xa, ya = sub5(NX, Ny, seed, tail)
        Xbb, ybb = sub5(NX, Ny, seed + 1000, tail)
        bedr.append(accuracy_score(yte, judge(jname, seed).fit(Xb, yb).predict(Xte)))
        na.append(accuracy_score(yte, judge(jname, seed).fit(Xa, ya).predict(Xte)))
        nb.append(accuracy_score(yte, judge(jname, seed).fit(Xbb, ybb).predict(Xte)))
    d1 = np.array(bedr) - np.array(na)
    d2 = np.array(na) - np.array(nb)
    results[jname] = {
        "BEDR_mean": round(float(np.mean(bedr)), 4), "BEDR_std": round(float(np.std(bedr)), 4),
        "NAIVE_mean": round(float(np.mean(na)), 4), "NAIVE_std": round(float(np.std(na)), 4),
        "delta_BEDR_minus_NAIVE_mean": round(float(d1.mean()), 4), "delta_std": round(float(d1.std()), 4), "frac_pos": round(float((d1 > 0).mean()), 3),
        "control_NAIVE_A_minus_B_mean": round(float(d2.mean()), 4), "control_std": round(float(d2.std()), 4), "control_frac_pos": round(float((d2 > 0).mean()), 3),
    }
    print(jname, json.dumps(results[jname]), flush=True)
json.dump({"tail_classes": [classes_sorted[c] for c in tail], "results": results}, open(os.path.join(OUT, "tail_flicker.json"), "w"), indent=1)
print("DONE", flush=True)

