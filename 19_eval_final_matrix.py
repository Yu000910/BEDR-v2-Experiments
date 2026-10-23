import numpy as np, json, os, time
from collections import Counter
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/final_matrix/", exist_ok=True)

OUT = os.path.join(BASE, "out/final_matrix")
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
cohort = json.load(open(BASE + "/out/fix_pilot/tail_cohort.json"))["tail"]

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

def load(path):
    z = np.load(path, allow_pickle=True)
    X = norm(z["vectors"].astype(np.float32))
    y_raw = np.asarray(z["labels"])
    if y_raw.dtype.kind in "UO":
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y

datasets = {}
datasets["HYBRID"] = load(BASE + "/out/round5/D_bedrv2_HYBRID.npz")
datasets["NONE"] = load(BASE + "/out/round5/D_bedrv2_NONE.npz")
datasets["PLAIN"] = load(BASE + "/out/round5/D_bedrv2_PLAIN.npz")
datasets["NAIVE"] = load(BASE + "/data/D_naive_full.npz")
datasets["PAPER"] = load(BASE + "/data/D_bedr_train.npz")
for name, (X, y) in datasets.items():
    cnt = Counter(y.tolist())
    vals = sorted(cnt.values())
    under30 = [classes_sorted[c] for c in cnt if cnt[c] < 30]
    print(name, X.shape[0], "rows,", len(cnt), "classes, min", vals[0], "median", vals[len(vals)//2], "max", vals[-1], "under30:", len(under30), flush=True)

zB = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
mB = np.asarray(zB["src"]) == "real"
XB = norm(zB["vectors"][mB].astype(np.float32)); yB = np.asarray(zB["labels"], dtype=np.int64)[mB]
zA = np.load(BASE + "/data/fullcov4291_plain.npz", allow_pickle=True)
XA = norm(zA["vectors"].astype(np.float32)); yA = np.asarray(zA["labels"], dtype=np.int64)
print("surface B:", XB.shape, "classes", len(set(yB.tolist())), "| surface A:", XA.shape, "classes", len(set(yA.tolist())), flush=True)

def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
def ridge(seed):
    return RidgeClassifier(alpha=1.0, random_state=seed)

def eval_surface(Xte, yte, judge_name, mode, k):
    results = {}
    for name, (X, y) in datasets.items():
        if mode == "matched":
            cnt = Counter(y.tolist())
            elig = sorted([c for c in set(y.tolist()) if cnt[c] >= k and c in set(yte.tolist())])
            mask = np.isin(yte, elig)
            Xe, ye = Xte[mask], yte[mask]
            rng = np.random.RandomState(0)
            accs = []
            for seed in [42, 43, 44, 45, 46]:
                parts = []
                for c in elig:
                    ci = np.where(y == c)[0]
                    parts.append(rng.choice(ci, size=k, replace=False))
                idx = np.concatenate(parts)
                clf = (mlp(seed) if judge_name == "MLP" else ridge(seed)).fit(X[idx], y[idx])
                accs.append(accuracy_score(ye, clf.predict(Xe)))
        else:
            Xe, ye = Xte, yte
            accs = []
            for seed in [42, 43, 44, 45, 46]:
                clf = (mlp(seed) if judge_name == "MLP" else ridge(seed)).fit(X, y)
                accs.append(accuracy_score(ye, clf.predict(Xe)))
        results[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
    return results

report = {}
for surf, (Xte, yte) in [("B", (XB, yB)), ("A", (XA, yA))]:
    for mode, k in [("full", None), ("matched", 30)]:
        for judge in ["MLP", "RIDGE"]:
            key = "S%s_%s_%s" % (surf, mode, judge)
            t0 = time.time()
            res = eval_surface(Xte, yte, judge, mode, k)
            report[key] = res
            print(key, "%.0fs" % (time.time() - t0), json.dumps(res), flush=True)
            json.dump(report, open(OUT + "/final_matrix.json", "w"), indent=1)

# paired t-test HYBRID vs NAIVE per key needs per-seed arrays; recompute quick for the main cells
def per_seed_accs(ds_name, judge_name, mode, k, Xte, yte):
    X, y = datasets[ds_name]
    if mode == "matched":
        cnt = Counter(y.tolist())
        elig = sorted([c for c in set(y.tolist()) if cnt[c] >= k and c in set(yte.tolist())])
        mask = np.isin(yte, elig)
        Xe, ye = Xte[mask], yte[mask]
        out = []
        for seed in [42, 43, 44, 45, 46]:
            rng = np.random.RandomState(seed)
            parts = []
            for c in elig:
                ci = np.where(y == c)[0]
                parts.append(rng.choice(ci, size=k, replace=False))
            idx = np.concatenate(parts)
            clf = (mlp(seed) if judge_name == "MLP" else ridge(seed)).fit(X[idx], y[idx])
            out.append(accuracy_score(ye, clf.predict(Xe)))
        return np.array(out), len(elig)
    else:
        Xe, ye = Xte, yte
        out = []
        for seed in [42, 43, 44, 45, 46]:
            clf = (mlp(seed) if judge_name == "MLP" else ridge(seed)).fit(X, y)
            out.append(accuracy_score(ye, clf.predict(Xe)))
        return np.array(out), len(set(y.tolist()))

stats_out = {}
for surf, (Xte, yte) in [("B", (XB, yB)), ("A", (XA, yA))]:
    for mode, k in [("full", None), ("matched", 30)]:
        for judge in ["MLP", "RIDGE"]:
            a, ea = per_seed_accs("HYBRID", judge, mode, k, Xte, yte)
            b, eb = per_seed_accs("NAIVE", judge, mode, k, Xte, yte)
            t, p = stats.ttest_rel(a, b)
            key = "S%s_%s_%s" % (surf, mode, judge)
            stats_out[key] = {"HYBRID": round(float(a.mean()), 4), "NAIVE": round(float(b.mean()), 4), "delta": round(float(a.mean() - b.mean()), 4), "t": round(float(t), 2), "p": float(p), "n_eligible": ea}
            print("TTEST", key, json.dumps(stats_out[key]), flush=True)
json.dump(stats_out, open(OUT + "/ttests.json", "w"), indent=1)

# tail subset on surface B with full MLP + ridge
tmask = np.isin(yB, cohort)
Xt, yt = XB[tmask], yB[tmask]
print("tail rows:", Xt.shape[0], flush=True)
tail_out = {}
for judge in ["MLP", "RIDGE"]:
    row = {}
    for name, (X, y) in datasets.items():
        accs = []
        for seed in [42, 43, 44, 45, 46]:
            clf = (mlp(seed) if judge == "MLP" else ridge(seed)).fit(X, y)
            accs.append(accuracy_score(yt, clf.predict(Xt)))
        row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
    tail_out[judge] = row
    print("TAIL", judge, json.dumps(row), flush=True)
json.dump(tail_out, open(OUT + "/tail.json", "w"), indent=1)
print("DONE", flush=True)

