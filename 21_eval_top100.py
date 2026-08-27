import numpy as np, json, os, time
from collections import Counter
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE + "/out/round5"
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
cohort = json.load(open(BASE + "/out/fix_pilot/tail_cohort.json"))["tail"]

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

z = np.load(OUT + "/D_bedrv2_HYBRID.npz", allow_pickle=True)
XP = norm(z["vectors"].astype(np.float32)); yP = np.asarray(z["labels"], dtype=np.int64)
zint = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint["labels"], zint["vectors"])}
XH = norm(XP + np.stack([intent_by_name[classes_sorted[c]] for c in yP]))

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

keep = []
for c in range(679):
    ci = np.where(yP == c)[0]
    if len(ci) <= 100:
        keep.append(ci)
    else:
        order = np.argsort(h[ci])[:100]
        keep.append(ci[order])
idx = np.concatenate(keep)
Xb2, yb2 = XP[idx], yP[idx]
np.savez(OUT + "/D_bedrv2_HYBRID_top100.npz", vectors=Xb2, labels=yb2)
print("top100 rows:", len(yb2), flush=True)

zn = np.load(BASE + "/data/D_naive_full.npz", allow_pickle=True)
XN = norm(zn["vectors"].astype(np.float32)); yN = np.array([name2idx[str(t)] for t in zn["labels"]], dtype=np.int64)
zB = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
mB = np.asarray(zB["src"]) == "real"
XB = norm(zB["vectors"][mB].astype(np.float32)); yB = np.asarray(zB["labels"], dtype=np.int64)[mB]
zA = np.load(BASE + "/data/orig4291_plain.npz", allow_pickle=True)
XA = norm(zA["vectors"].astype(np.float32)); yA = np.asarray(zA["labels"], dtype=np.int64)

def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
def ridge(seed):
    return RidgeClassifier(alpha=1.0, random_state=seed)

results = {}
for surf, (Xte, yte) in [("B", (XB, yB)), ("A", (XA, yA))]:
    for judge in ["MLP", "RIDGE"]:
        for name, (X, y) in [("TOP100", (Xb2, yb2)), ("NAIVE", (XN, yN))]:
            accs = []
            for seed in [42, 43, 44, 45, 46]:
                clf = (mlp(seed) if judge == "MLP" else ridge(seed)).fit(X, y)
                accs.append(accuracy_score(yte, clf.predict(Xte)))
            results["S%s_%s_%s" % (surf, judge, name)] = accs
            print("S%s %s %s: %.4f +-%.4f" % (surf, judge, name, np.mean(accs), np.std(accs)), flush=True)
        a = np.array(results["S%s_%s_TOP100" % (surf, judge)])
        b = np.array(results["S%s_%s_NAIVE" % (surf, judge)])
        t, p = stats.ttest_rel(a, b)
        print("  t-test TOP100 vs NAIVE: delta %+.4f t=%.2f p=%.4g" % (a.mean() - b.mean(), t, p), flush=True)

tmask = np.isin(yB, cohort)
Xt, yt = XB[tmask], yB[tmask]
for judge in ["MLP", "RIDGE"]:
    row = {}
    for name, (X, y) in [("TOP100", (Xb2, yb2)), ("NAIVE", (XN, yN))]:
        accs = []
        for seed in [42, 43, 44, 45, 46]:
            clf = (mlp(seed) if judge == "MLP" else ridge(seed)).fit(X, y)
            accs.append(accuracy_score(yt, clf.predict(Xt)))
        row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
    print("TAIL", judge, json.dumps(row), flush=True)
json.dump({k: [round(float(x), 4) for x in v] for k, v in results.items()}, open(OUT + "/top100_matrix.json", "w"), indent=1)
print("DONE", flush=True)

