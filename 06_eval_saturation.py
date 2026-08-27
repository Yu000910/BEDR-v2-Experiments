import numpy as np, json, os
from collections import Counter
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5", exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zn = np.load(BASE + "/data/D_naive_full.npz", allow_pickle=True)
XN = norm(zn["vectors"]); yN = np.array([name2idx[str(t)] for t in zn["labels"]], dtype=np.int64)
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XT = norm(z2["vectors"][m]); yT = np.asarray(z2["labels"], dtype=np.int64)[m]

cnt = Counter(yN.tolist())
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

def matched(k, seed, elig):
    rng = np.random.RandomState(seed)
    parts = []
    for c in elig:
        ci = np.where(yN == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False))
    idx = np.concatenate(parts)
    return XN[idx], yN[idx]

print("%-4s %-8s %-16s %-16s %-12s" % ("k", "elig", "NC acc", "Ridge acc", "NC marginal"), flush=True)
prev_nc = None
out = {}
for k in [5, 10, 15, 20, 30, 40, 50]:
    elig = sorted([c for c in cnt if cnt[c] >= k and c in set(yT.tolist())])
    mask = np.isin(yT, elig)
    Xte, yte = XT[mask], yT[mask]
    nc_accs, ridge_accs = [], []
    for seed in [42, 43, 44]:
        Xs, ys = matched(k, seed, elig)
        nc_accs.append(accuracy_score(yte, CosNC().fit(Xs, ys).predict(Xte)))
        ridge_accs.append(accuracy_score(yte, RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys).predict(Xte)))
    nc = float(np.mean(nc_accs)); ridge = float(np.mean(ridge_accs))
    marg = round(nc - prev_nc, 4) if prev_nc is not None else None
    out["k%d" % k] = {"elig": len(elig), "NC": round(nc, 4), "Ridge": round(ridge, 4), "NC_marginal": marg}
    print("%-4d %-8d %-16s %-16s %-12s" % (k, len(elig), round(nc, 4), round(ridge, 4), marg), flush=True)
    prev_nc = nc
json.dump(out, open(BASE + "/out/round5/saturation.json", "w"), indent=1)
print("DONE", flush=True)

