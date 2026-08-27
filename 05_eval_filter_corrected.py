import numpy as np, json, os, time
from collections import Counter
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/filter_corrected/", exist_ok=True)

OUT = os.path.join(BASE, "out/filter_corrected")
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(os.path.join(BASE, "data/classes_sorted.json")))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zb = np.load(os.path.join(BASE, "data/D_base_train_new.npz"), allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([name2idx[str(t)] for t in zb["labels"]], dtype=np.int64)
print("base:", XB.shape, flush=True)

emb = np.load(os.path.join(BASE, "data/audit/embeds.npz"))
txts = json.load(open(os.path.join(BASE, "data/audit/texts.json")))
rem_labels = [name2idx[t] for t, _ in txts["removed"]]
kept_labels = [name2idx[t] for t, _ in txts["kept"]]
XC = np.vstack([norm(emb["rem_emb"]), norm(emb["kept_emb"])])
yC = np.array(rem_labels + kept_labels, dtype=np.int64)
is_removed = np.array([True] * len(rem_labels) + [False] * len(kept_labels))
print("candidates:", XC.shape, "removed:", int(is_removed.sum()), flush=True)

z2 = np.load(os.path.join(BASE, "data/test2_full_plain.npz"), allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XT = norm(z2["vectors"][m]); yT = np.asarray(z2["labels"], dtype=np.int64)[m]
zn = np.load(os.path.join(BASE, "data/D_naive_full.npz"), allow_pickle=True)
XN = norm(zn["vectors"]); yN = np.array([name2idx[str(t)] for t in zn["labels"]], dtype=np.int64)
zbedr = np.load(os.path.join(BASE, "data/D_bedr_train.npz"), allow_pickle=True)
XBedr = norm(zbedr["vectors"]); yBedr = np.array([name2idx[str(t)] for t in zbedr["labels"]], dtype=np.int64)
print("test B:", XT.shape, "naive:", XN.shape, "paper_bedr:", XBedr.shape, flush=True)

XP = np.vstack([XB, XC]); yP = np.concatenate([yB, yC])
nB, nC, nP = len(yB), len(yC), len(yP)
K, TOP = 15, 25

t0 = time.time()
top_idx = np.zeros((nP, TOP), dtype=np.int64)
top_dst = np.zeros((nP, TOP), dtype=np.float32)
BS = 2048
for i in range(0, nP, BS):
    D = 1.0 - XP[i:i + BS] @ XP.T
    D[i:i + BS, i:i + BS] = 1.0
    part = np.argpartition(D, TOP - 1, axis=1)[:, :TOP]
    top_idx[i:i + BS] = part
    top_dst[i:i + BS] = np.take_along_axis(D, part, axis=1)
print("top25 done", round(time.time() - t0, 1), "s", flush=True)

def entropy_of(idx_list):
    labs = yP[idx_list]
    vals, counts = np.unique(labs, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

t0 = time.time()
h = np.zeros(nP, dtype=np.float32)
for i in range(nP):
    h[i] = entropy_of(top_idx[i, :K])
print("local entropy done", round(time.time() - t0, 1), "s", flush=True)

t0 = time.time()
deltas = np.zeros(nC, dtype=np.float32)
for ci in range(nC):
    c = nB + ci
    delta = h[c]
    aff = np.where((top_idx[:, :TOP] == c).any(axis=1))[0]
    aff = aff[aff != c]
    dcj = 1.0 - XP[aff] @ XP[c]
    for jj, j in enumerate(aff):
        cand = np.concatenate([top_idx[j, :TOP], np.array([c])])
        dd = np.concatenate([top_dst[j, :TOP], np.array([dcj[jj]])])
        order = np.argsort(dd)[:K]
        delta += entropy_of(cand[order]) - h[j]
    deltas[ci] = delta
    if (ci + 1) % 2000 == 0:
        print("deltas", ci + 1, "/", nC, round(time.time() - t0, 1), "s", flush=True)
print("marginal deltas done in", round(time.time() - t0, 1), "s", flush=True)

accept_global = deltas < 0
accept_local10 = h[nB:] < 1.0
accept_local23 = h[nB:] < 2.3
print("accept GLOBAL(delta<0):", int(accept_global.sum()), flush=True)
print("accept LOCAL<1.0:", int(accept_local10.sum()), flush=True)
print("accept LOCAL<2.3:", int(accept_local23.sum()), flush=True)
print("delta mean %.4f median %.4f p10 %.4f p90 %.4f" % (deltas.mean(), np.median(deltas), np.percentile(deltas, 10), np.percentile(deltas, 90)), flush=True)
np.savez(os.path.join(OUT, "filter_decision.npz"), deltas=deltas, h_local=h[nB:], accept_global=accept_global, accept_local10=accept_local10, accept_local23=accept_local23, is_removed=is_removed)

datasets = {
    "UNF_all": (np.vstack([XB, XC]), np.concatenate([yB, yC])),
    "F_GLOBAL": (np.vstack([XB, XC[accept_global]]), np.concatenate([yB, yC[accept_global]])),
    "F_LOC10": (np.vstack([XB, XC[accept_local10]]), np.concatenate([yB, yC[accept_local10]])),
    "F_LOC23": (np.vstack([XB, XC[accept_local23]]), np.concatenate([yB, yC[accept_local23]])),
    "PAPER": (XBedr, yBedr),
    "NAIVE": (XN, yN),
}
for name, (X, y) in datasets.items():
    print(name, "rows", X.shape[0], "classes", len(set(y.tolist())), flush=True)

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
    if name == "RIDGE": return RidgeClassifier(alpha=1.0, random_state=seed)
    if name == "SGD": return SGDClassifier(loss="log_loss", max_iter=30, alpha=1e-4, random_state=seed, n_jobs=-1)
    raise ValueError(name)

def matched(X, y, k, seed, classes):
    rng = np.random.RandomState(seed)
    parts = []
    for c in classes:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False))
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

results = {}
for k in [5, 10, 30]:
    counts = {}
    for name, (X, y) in datasets.items():
        cnt = Counter(y.tolist())
        counts[name] = cnt
    elig = set(counts["UNF_all"].keys())
    for name in datasets:
        elig &= set(counts[name].keys())
    elig = [c for c in sorted(elig) if all(counts[nm][c] >= k for nm in datasets) and c in set(yT.tolist())]
    print("k=%d eligible=%d" % (k, len(elig)), flush=True)
    mask = np.isin(yT, elig)
    Xte, yte = XT[mask], yT[mask]
    for jname in ["NC", "RIDGE", "SGD"]:
        row = {}
        for name, (X, y) in datasets.items():
            accs = []
            for seed in [42, 43, 44]:
                Xs, ys = matched(X, y, k, seed, elig)
                clf = judge(jname, seed).fit(Xs, ys)
                accs.append(accuracy_score(yte, clf.predict(Xte)))
            row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
        key = "k%d|%s" % (k, jname)
        results[key] = row
        print(key, json.dumps(row), flush=True)

json.dump({"results": results, "eligible_notes": "classes with >=k in all six datasets and in surface-B test"}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("DONE", flush=True)

