import numpy as np, json, os
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

OUT = os.path.join(BASE, "out/round5")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
cohort = json.load(open(BASE + "/out/fix_pilot/tail_cohort.json"))["tail"]

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

def load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    X = norm(z["vectors"])
    y_raw = np.asarray(z["labels"])
    if y_raw.dtype.kind in "UO":
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XT = norm(z2["vectors"][m]); yT = np.asarray(z2["labels"], dtype=np.int64)[m]

datasets = {}
for arm in ["NONE", "PLAIN", "HYBRID"]:
    datasets["R5_" + arm] = load(OUT + "/D_round5_" + arm + ".npz")
datasets["NAIVE"] = load(BASE + "/data/D_naive_full.npz")
datasets["PAPER"] = load(BASE + "/data/D_bedr_train.npz")
for name, (X, y) in datasets.items():
    print(name, X.shape[0], "rows,", len(set(y.tolist())), "classes", flush=True)

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
    if name == "SGD": return SGDClassifier(loss="log_loss", max_iter=20, alpha=1e-4, random_state=seed, n_jobs=-1)
    if name == "RIDGE": return RidgeClassifier(alpha=1.0, random_state=seed)
    if name == "MLP": return MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
    raise ValueError(name)

def matched(X, y, k, seed, classes):
    rng = np.random.RandomState(seed)
    parts = []
    for c in classes:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False) if len(ci) >= k else ci)
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

results = {}
counts = {name: Counter(y.tolist()) for name, (X, y) in datasets.items()}
for k in [5, 10, 30]:
    elig = set(counts["NAIVE"].keys())
    for name in datasets:
        elig &= set(counts[name].keys())
    elig = sorted([c for c in elig if all(counts[nm][c] >= k for nm in datasets) and c in set(yT.tolist())])
    print("k=%d eligible=%d" % (k, len(elig)), flush=True)
    judges_k = ["NC", "KNN1", "KNN5", "SGD", "RIDGE"] + (["MLP"] if k == 30 else [])
    for jname in judges_k:
        row = {}
        for name, (X, y) in datasets.items():
            accs = []
            for seed in [42, 43, 44]:
                Xs, ys = matched(X, y, k, seed, elig)
                mask = np.isin(yT, elig)
                clf = judge(jname, seed).fit(Xs, ys)
                accs.append(accuracy_score(yT[mask], clf.predict(XT[mask])))
            row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
        results["k%d|%s" % (k, jname)] = row
        print("k%d|%s" % (k, jname), json.dumps(row), flush=True)

# tail subset: cohort classes, NC/Ridge at k=min(10)
print("=== tail subset ===", flush=True)
tmask = np.isin(yT, cohort)
Xtt, ytt = XT[tmask], yT[tmask]
print("tail test rows:", Xtt.shape[0], flush=True)
for jname in ["NC", "RIDGE"]:
    row = {}
    for name, (X, y) in datasets.items():
        accs = []
        for seed in [42, 43, 44]:
            Xs, ys = matched(X, y, 10, seed, cohort)
            clf = judge(jname, seed).fit(Xs, ys)
            accs.append(accuracy_score(ytt, clf.predict(Xtt)))
        row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
    results["tail|" + jname] = row
    print("tail|" + jname, json.dumps(row), flush=True)

json.dump(results, open(OUT + "/judge_results.json", "w"), indent=1)
print("DONE", flush=True)

