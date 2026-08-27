import numpy as np, json, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/diversity/", exist_ok=True)

OUT = os.path.join(BASE, "out/diversity")
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

def matched_subsample(X, y, k, seed, classes):
    rng = np.random.RandomState(seed)
    out = []
    for c in classes:
        ci = np.where(y == c)[0]
        out.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(out)

def per_class_stats(Xsel, ysel, k):
    recs = {}
    for c in sorted(set(ysel.tolist())):
        ci = np.where(ysel == c)[0]
        Xc = Xsel[ci]
        S = Xc @ Xc.T
        ii, jj = np.triu_indices(k, k=1)
        pair = S[ii, jj]
        cent = Xc.mean(axis=0)
        cn = cent / (np.linalg.norm(cent) + 1e-12)
        d2c = Xc @ cn
        recs[c] = {"mean_pair_cos": float(pair.mean()), "max_pair_cos": float(pair.max()), "mean_centroid_cos": float(d2c.mean())}
    return recs

report = {}
for k in [10, 30]:
    cb = {c: int((By == c).sum()) for c in set(By.tolist())}
    cn = {c: int((Ny == c).sum()) for c in set(Ny.tolist())}
    elig = sorted([c for c in set(cb) & set(cn) if cb[c] >= k and cn[c] >= k])
    agg = {"mean_pair_cos": {"B": [], "N": []}, "max_pair_cos": {"B": [], "N": []}, "mean_centroid_cos": {"B": [], "N": []}}
    per_seed = []
    for seed in [42, 43, 44]:
        idx_b = matched_subsample(BX, By, k, seed, elig)
        idx_n = matched_subsample(NX, Ny, k, seed, elig)
        rb = per_class_stats(BX[idx_b], By[idx_b], k)
        rn = per_class_stats(NX[idx_n], Ny[idx_n], k)
        per_seed.append((rb, rn))
    for c in elig:
        for metric in agg:
            agg[metric]["B"].append(float(np.mean([s[0][c][metric] for s in per_seed])))
            agg[metric]["N"].append(float(np.mean([s[1][c][metric] for s in per_seed])))
    print("k=%d eligible classes=%d" % (k, len(elig)), flush=True)
    for metric in agg:
        b = np.array(agg[metric]["B"]); n = np.array(agg[metric]["N"])
        print("  %-18s BEDR %.4f +-%.4f | NAIVE %.4f +-%.4f | delta %+.4f | frac(B>N)=%.3f" % (metric, b.mean(), b.std(), n.mean(), n.std(), (b - n).mean(), (b > n).mean()), flush=True)
    report["k%d" % k] = {"eligible": elig, "BEDR": agg["mean_pair_cos"]["B"], "NAIVE": agg["mean_pair_cos"]["N"]}

print("=== real-test coverage (k=30, seed 42) ===", flush=True)
for name, (X, y) in [("BEDR", (BX, By)), ("NAIVE", (NX, Ny))]:
    rng = np.random.RandomState(42)
    cent = {}
    cb = {c: int((By == c).sum()) for c in set(By.tolist())}
    cn = {c: int((Ny == c).sum()) for c in set(Ny.tolist())}
    elig = [c for c in set(cb) & set(cn) if cb[c] >= 30 and cn[c] >= 30]
    for c in elig:
        ci = np.where(y == c)[0]
        ci = rng.choice(ci, size=30, replace=False)
        cent[c] = X[ci].mean(axis=0)
    per_class = defaultdict(list)
    for row, c in zip(TX, Ty.tolist()):
        if c in cent:
            cnv = cent[c] / (np.linalg.norm(cent[c]) + 1e-12)
            per_class[c].append(float(row @ cnv))
    means = [float(np.mean(v)) for v in per_class.values()]
    print("%-6s classes=%d mean test-to-centroid cos %.4f" % (name, len(means), float(np.mean(means))), flush=True)

json.dump(report, open(os.path.join(OUT, "diversity.json"), "w"))
print("DONE", flush=True)

