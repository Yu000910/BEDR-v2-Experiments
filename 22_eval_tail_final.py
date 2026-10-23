import numpy as np, json
from collections import Counter
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
cohort = json.load(open(BASE + "/out/fix_pilot/tail_cohort.json"))["tail"]

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

def load(path, kind):
    z = np.load(path, allow_pickle=True)
    X = norm(z["vectors"].astype(np.float32))
    y_raw = np.asarray(z["labels"])
    if y_raw.dtype.kind in "UO":
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y

Xb, yb = load(BASE + "/out/round5/D_bedr_HYBRID_top100.npz", "int")
Xn, yn = load(BASE + "/data/D_naive_full.npz", "name")
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XT = norm(z2["vectors"][m].astype(np.float32)); yT = np.asarray(z2["labels"], dtype=np.int64)[m]
tmask = np.isin(yT, cohort)
Xtt, ytt = XT[tmask], yT[tmask]
print("tail test rows:", Xtt.shape[0], flush=True)

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

def matched(X, y, k, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for c in cohort:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False) if len(ci) >= k else ci)
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

for k in [10, 20, 30]:
    for jname, fitter in [("NC", CosNC), ("RIDGE", lambda s: RidgeClassifier(alpha=1.0, random_state=s))]:
        for name, (X, y) in [("BEDR", (Xb, yb)), ("NAIVE", (Xn, yn))]:
            accs = []
            for seed in [42, 43, 44]:
                Xs, ys = matched(X, y, k, seed)
                clf = (fitter() if jname == "NC" else fitter(seed)).fit(Xs, ys)
                accs.append(accuracy_score(ytt, clf.predict(Xtt)))
            print("k=%d %s %s: %.4f +-%.4f" % (k, jname, name, np.mean(accs), np.std(accs)), flush=True)
print("DONE", flush=True)

