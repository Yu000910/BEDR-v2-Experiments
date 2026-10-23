import numpy as np, json, os, time
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE + "/out/round5"
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

z = np.load(OUT + "/D_bedr_HYBRID.npz", allow_pickle=True)
XP = norm(z["vectors"].astype(np.float32)); yP = np.asarray(z["labels"], dtype=np.int64)
zint = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint["labels"], zint["vectors"])}
XH = norm(XP + np.stack([intent_by_name[classes_sorted[c]] for c in yP]))
print("pool:", XP.shape, flush=True)

t0 = time.time()
h = np.zeros(len(yP), dtype=np.float32)
BS = 2048
TOP = 16
for i in range(0, len(yP), BS):
    D = 1.0 - XH[i:i + BS] @ XH.T
    D[i:i + BS, i:i + BS] = 1.0
    part = np.argpartition(D, TOP - 1, axis=1)[:, :TOP]
    idx = part[:, :15]
    labs = yP[idx]
    for r in range(len(idx)):
        vals, counts = np.unique(labs[r], return_counts=True)
        p = counts / counts.sum()
        h[i + r] = -(p * np.log2(p)).sum()
print("hybrid local entropy done", round(time.time() - t0, 1), "s", flush=True)

cnt = Counter(yP.tolist())
heads = [c for c in cnt if cnt[c] > 80]
print("head classes (>80):", len(heads), "samples above 80:", sum(cnt[c] - 80 for c in heads), flush=True)

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XB = norm(z2["vectors"][m].astype(np.float32)); yB = np.asarray(z2["labels"], dtype=np.int64)[m]

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

for TOPN in [80, 100]:
    keep = []
    for c in range(679):
        ci = np.where(yP == c)[0]
        if len(ci) <= TOPN:
            keep.append(ci)
        else:
            order = np.argsort(h[ci])[:TOPN]
            keep.append(ci[order])
    idx = np.concatenate(keep)
    Xs, ys = XP[idx], yP[idx]
    print("TOPN=%d rows=%d" % (TOPN, len(ys)), flush=True)
    nc_accs, ridge_accs, mlp_accs = [], [], []
    for seed in [42, 43, 44]:
        nc_accs.append(accuracy_score(yB, CosNC().fit(Xs, ys).predict(XB)))
        ridge_accs.append(accuracy_score(yB, RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys).predict(XB)))
        mlp_accs.append(accuracy_score(yB, MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed).fit(Xs, ys).predict(XB)))
    print("  NC %.4f | Ridge %.4f | MLP %.4f +-%.4f" % (np.mean(nc_accs), np.mean(ridge_accs), np.mean(mlp_accs), np.std(mlp_accs)), flush=True)
print("DONE", flush=True)

