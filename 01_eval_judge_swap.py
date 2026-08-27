import numpy as np, json, os, time
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/judge_swap/", exist_ok=True)

OUTDIR = os.path.join(BASE, "out/judge_swap")
os.makedirs(OUTDIR, exist_ok=True)

classes_sorted = json.load(open(os.path.join(BASE, "data/classes_sorted.json")))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def load(path, kind):
    z = np.load(path, allow_pickle=True)
    X = z["vectors"].astype(np.float32)
    if kind == "name":
        y = np.array([name2idx[str(t)] for t in z["labels"]], dtype=np.int64)
    else:
        y = np.asarray(z["labels"], dtype=np.int64)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n, y

print("loading...", flush=True)
BX, By = load(os.path.join(BASE, "data/D_bedr_train.npz"), "name")
NX, Ny = load(os.path.join(BASE, "data/D_naive_full.npz"), "name")
TA_X, TA_y = load(os.path.join(BASE, "data/orig4291_plain.npz"), "int")
z2 = np.load(os.path.join(BASE, "data/test2_full_plain.npz"), allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
TB_X = z2["vectors"][m].astype(np.float32)
TB_y = np.asarray(z2["labels"], dtype=np.int64)[m]
nn = np.linalg.norm(TB_X, axis=1, keepdims=True); nn[nn == 0] = 1.0
TB_X = TB_X / nn
assert set(TB_y.tolist()) <= set(range(len(classes_sorted))), "test2 labels out of range"
print("surface A (orig test):", TA_X.shape, "classes", len(set(TA_y.tolist())), flush=True)
print("surface B (real-only):", TB_X.shape, "classes", len(set(TB_y.tolist())), flush=True)

cb, cn = Counter(By.tolist()), Counter(Ny.tolist())
def stats(c):
    vals = list(c.values())
    return {"n_classes": len(vals), "min": min(vals), "median": int(np.median(vals)), "mean": round(float(np.mean(vals)),1), "max": max(vals), "lt10": sum(1 for v in vals if v < 10), "lt20": sum(1 for v in vals if v < 20)}
print("BEDR train:", BX.shape, stats(cb), flush=True)
print("NAIVE train:", NX.shape, stats(cn), flush=True)

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
    if name == "SGD": return SGDClassifier(loss="log_loss", max_iter=10, alpha=1e-4, random_state=seed, n_jobs=-1)
    if name == "RIDGE": return RidgeClassifier(alpha=1.0, random_state=seed)
    if name == "MLP": return MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
    raise ValueError(name)

def subsample(X, y, k, seed, eligible):
    rng = np.random.RandomState(seed)
    parts = []
    for c in eligible:
        ci = np.where(y == c)[0]
        if len(ci) > k:
            ci = rng.choice(ci, size=k, replace=False)
        parts.append(ci)
    idx = np.concatenate(parts)
    return X[idx], y[idx]

JUDGES = ["NC", "KNN1", "KNN5", "SGD", "RIDGE", "MLP"]
SEEDS = [42, 43, 44]
results = {}
meta = {}

# ref-full on surface A
ref = {}
ctA = set(TA_y.tolist())
for dsname, (X, y) in [("BEDR", (BX, By)), ("NAIVE", (NX, Ny))]:
    elig = sorted(set(y.tolist()) & ctA)
    mask = np.isin(TA_y, elig)
    Xte, yte = TA_X[mask], TA_y[mask]
    for jname in ["NC", "KNN5", "SGD", "RIDGE", "MLP"]:
        t0 = time.time()
        clf = judge(jname, 42).fit(X, y)
        acc = accuracy_score(yte, clf.predict(Xte))
        ref[dsname + "|" + jname] = {"acc": round(float(acc),4), "n_train": int(X.shape[0]), "n_test": int(Xte.shape[0]), "secs": round(time.time()-t0,1)}
        print("REF-A", dsname, jname, round(float(acc),4), flush=True)
json.dump(ref, open(os.path.join(OUTDIR, "ref_full.json"), "w"), indent=1)

SURFACES = [("A", TA_X, TA_y), ("B", TB_X, TB_y)]
for surf, TX, Ty in SURFACES:
    cts = set(Ty.tolist())
    ks = [5, 10, 30] if surf == "B" else [30]
    for k in ks:
        elig = [c for c in sorted(set(cb.keys()) & set(cn.keys())) if cb[c] >= k and cn[c] >= k and c in cts]
        tail5 = [c for c in elig if cb[c] < 10]
        tail20 = [c for c in elig if cb[c] < 20]
        meta["S%s_k%d" % (surf, k)] = {"n_eligible": len(elig), "n_tail_lt10": len(tail5), "n_tail_lt20": len(tail20)}
        print("S%s k=%d eligible=%d tail(<10)=%d tail(<20)=%d" % (surf, k, len(elig), len(tail5), len(tail20)), flush=True)
        for dsname, (X, y) in [("BEDR", (BX, By)), ("NAIVE", (NX, Ny))]:
            for jname in JUDGES:
                accs, macro_accs, t5, t20 = [], [], [], []
                for seed in SEEDS:
                    Xs, ys = subsample(X, y, k, seed, elig)
                    mask = np.isin(Ty, elig)
                    Xte, yte = TX[mask], Ty[mask]
                    clf = judge(jname, seed).fit(Xs, ys)
                    pred = clf.predict(Xte)
                    accs.append(accuracy_score(yte, pred))
                    per = {}
                    for c, p in zip(yte.tolist(), pred.tolist()):
                        per.setdefault(c, []).append(int(c == p))
                    macro_accs.append(float(np.mean([np.mean(v) for v in per.values()])))
                    if k == 5:
                        for tname, tset in [("lt10", tail5), ("lt20", tail20)]:
                            if len(tset) >= 5:
                                tmask = np.isin(Ty, tset)
                                if tmask.sum() > 0:
                                    tacc = accuracy_score(Ty[tmask], clf.predict(TX[tmask]))
                                    (t5 if tname == "lt10" else t20).append(tacc)
                key = "S%s_k%d|%s|%s" % (surf, k, dsname, jname)
                results[key] = {
                    "acc_mean": round(float(np.mean(accs)),4), "acc_std": round(float(np.std(accs)),4),
                    "macro_mean": round(float(np.mean(macro_accs)),4), "macro_std": round(float(np.std(macro_accs)),4),
                    "tail_lt10_acc": round(float(np.mean(t5)),4) if t5 else None,
                    "tail_lt20_acc": round(float(np.mean(t20)),4) if t20 else None,
                    "n_eligible": len(elig), "n_train_rows": k * len(elig),
                }
                print(key, json.dumps(results[key]), flush=True)
    json.dump(results, open(os.path.join(OUTDIR, "results.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(OUTDIR, "meta.json"), "w"), indent=1)

lines = []
for surf in ["A", "B"]:
    ks = [30] if surf == "A" else [5, 10, 30]
    for k in ks:
        mkey = "S%s_k%d" % (surf, k)
        lines.append("=== surface %s k=%d (eligible=%d, tail_lt10=%d) ===" % (surf, k, meta[mkey]["n_eligible"], meta[mkey]["n_tail_lt10"]))
        lines.append("%-6s | %-22s | %-22s | %-8s" % ("judge", "BEDR acc (macro)", "NAIVE acc (macro)", "winner"))
        for jname in JUDGES:
            b = results["S%s_k%d|BEDR|%s" % (surf, k, jname)]
            n = results["S%s_k%d|NAIVE|%s" % (surf, k, jname)]
            w = "BEDR" if b["acc_mean"] > n["acc_mean"] + 0.002 else ("NAIVE" if n["acc_mean"] > b["acc_mean"] + 0.002 else "TIE")
            lines.append("%-6s | %7.4f +-%.4f (%7.4f) | %7.4f +-%.4f (%7.4f) | %-8s" % (jname, b["acc_mean"], b["acc_std"], b["macro_mean"], n["acc_mean"], n["acc_std"], n["macro_mean"], w))
        if k == 5:
            for jname in JUDGES:
                b = results["S%s_k5|BEDR|%s" % (surf, jname)]; n = results["S%s_k5|NAIVE|%s" % (surf, jname)]
                lines.append("  tail(<10) %s: BEDR %.4f vs NAIVE %.4f ; tail(<20): %.4f vs %.4f" % (jname, b["tail_lt10_acc"] or -1, n["tail_lt10_acc"] or -1, b["tail_lt20_acc"] or -1, n["tail_lt20_acc"] or -1))
lines.append("=== REF full-data on surface A ===")
for jname in ["NC", "KNN5", "SGD", "RIDGE", "MLP"]:
    lines.append("%-6s BEDR %.4f vs NAIVE %.4f" % (jname, ref["BEDR|" + jname]["acc"], ref["NAIVE|" + jname]["acc"]))
open(os.path.join(OUTDIR, "summary.txt"), "w").write(chr(10).join(lines))
print("DONE", flush=True)

