# -*- coding: utf-8 -*-
"""37_gbe_error_analysis.py
边界熵有效性验证（对应第三章 3.3.3 节（9））：
  ① 标签集级全局边界熵（GBE）：基础集 / 朴素集 / 多轮自适应过采样后 / 终版
  ② 样本级：真实语句测试集上的局部边界熵（LBE）与错误率对应关系
空间：普通句向量（与训练、评测空间一致）；k=15；余弦距离；GBE 为数据集内部 LBE 的均值（自排除）。
输出：results/gbe_error.json
"""
import numpy as np, json, os
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
D = BASE + "/data"
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(D + "/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}


def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n


def load(p, kind="name"):
    z = np.load(p, allow_pickle=True)
    X = norm(z["vectors"].astype(np.float32))
    yr = z["labels"]
    y = np.array([n2i[str(t)] for t in yr], dtype=np.int64) if kind == "name" else yr.astype(np.int64)
    return X, y


def lbe_self(X, y, k=15, bs=2048):
    h = np.zeros(len(y), dtype=np.float32)
    for i in range(0, len(y), bs):
        Dm = 1.0 - X[i:i + bs] @ X.T
        r = np.arange(len(Dm)); Dm[r, i + r] = 1.0
        part = np.argpartition(Dm, k - 1, axis=1)[:, :k]; labs = y[part]
        for j in range(len(part)):
            v, c = np.unique(labs[j], return_counts=True); p = c / c.sum()
            h[i + j] = -(p * np.log2(p)).sum()
    return h


def lbe_ref(Xq, Xr, yr, k=15, bs=1024):
    h = np.zeros(len(Xq), dtype=np.float32)
    for i in range(0, len(Xq), bs):
        Dm = 1.0 - Xq[i:i + bs] @ Xr.T
        part = np.argpartition(Dm, k - 1, axis=1)[:, :k]; labs = yr[part]
        for j in range(len(part)):
            v, c = np.unique(labs[j], return_counts=True); p = c / c.sum()
            h[i + j] = -(p * np.log2(p)).sum()
    return h


out = {}

# ---------- ① 标签集级全局边界熵 ----------
sets = [("baseline", D + "/D_base_train_new.npz", "name"),
        ("naive", D + "/D_naive_full.npz", "name"),
        ("oversampled", D + "/D_arms_HYBRID.npz", "int"),
        ("final", D + "/D_final.npz", "int")]
gbe = {}
for tag, p, kind in sets:
    X, y = load(p, kind)
    gbe[tag] = {"n": int(len(y)), "classes": int(len(set(y.tolist()))),
                "gbe": round(float(lbe_self(X, y).mean()), 4)}
    print("GBE %-12s n=%6d classes=%3d  %.4f" % (tag, gbe[tag]["n"], gbe[tag]["classes"], gbe[tag]["gbe"]), flush=True)
out["gbe_plain"] = gbe

# ---------- ② 熵—错误率 ----------
XT, yT = load(D + "/D_final.npz", "int")
z2 = np.load(D + "/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XE = norm(z2["vectors"][m].astype(np.float32)); yE = np.asarray(z2["labels"], dtype=np.int64)[m]
h = lbe_ref(XE, XT, yT)
qs = [float(q) for q in np.quantile(h, [0, .25, .5, .75, 1.0])]
res = {"lbe_mean": round(float(h.mean()), 4), "lbe_median": round(float(np.median(h)), 4),
       "quartile_edges": [round(q, 4) for q in qs], "judges": {}}
for jname in ["MLP", "RIDGE"]:
    errs = []
    for seed in [42, 43, 44, 45, 46]:
        clf = (MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True,
                             n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
               if jname == "MLP" else RidgeClassifier(alpha=1.0, random_state=seed))
        clf.fit(XT, yT)
        errs.append((clf.predict(XE) != yE).astype(np.float64))
    err = np.mean(errs, axis=0)
    rows = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        mask = (h >= lo) & (h <= hi) if i == 3 else (h >= lo) & (h < hi)
        rows.append({"quartile": i + 1, "range": [round(lo, 3), round(hi, 3)],
                     "n": int(mask.sum()), "error": round(float(err[mask].mean()), 4)})
    lo_mask = h <= 2.0
    res["judges"][jname] = {
        "accuracy": round(float(1 - err.mean()), 4),
        "quartiles": rows,
        "corr_lbe_error": round(float(np.corrcoef(h, err)[0, 1]), 4),
        "low_entropy_le2": {"n": int(lo_mask.sum()), "error": round(float(err[lo_mask].mean()), 4)},
        "high_entropy_gt2": {"n": int((~lo_mask).sum()), "error": round(float(err[~lo_mask].mean()), 4)},
    }
    print(jname, json.dumps(res["judges"][jname], ensure_ascii=False), flush=True)

clf = RidgeClassifier(alpha=1.0, random_state=42).fit(XT, yT)
pred = clf.predict(XE)
rows = []
for c in np.unique(yE):
    mm = yE == c
    if mm.sum() < 3:
        continue
    rows.append((float(h[mm].mean()), float((pred[mm] == c).mean()), int(mm.sum())))
rows = np.array(rows)
res["class_level"] = {"n_classes": int(len(rows)),
                      "corr_class_lbe_accuracy": round(float(np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]), 4)}
print("class-level:", res["class_level"], flush=True)

json.dump(out | {"entropy_error": res}, open(BASE + "/results/gbe_error.json", "w"), indent=1, ensure_ascii=False)
print("DONE ->", BASE + "/results/gbe_error.json", flush=True)
