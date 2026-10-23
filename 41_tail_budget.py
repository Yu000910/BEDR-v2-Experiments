# -*- coding: utf-8 -*-
"""41_tail_budget.py
论文长尾等预算对照（表：322 个长尾类别 / 780 条真实测试句）。

cohort 定义（与论文 3.3.3 节完全一致）：
    基础集中样本数 < 30 的类别，且该类别在真实语句测试集中至少有一条测试句。
    实测：322 个类别、780 条测试句。

对照：优化集（D_final）vs 朴素集（D_naive_full），每类预算 k = 10 / 20 / 30，
      判据：最近质心（余弦）+ Ridge，3 个随机种子（42–44），报告均值 ± 标准差。
输出：results/tail_budget_322.json
用法：cd 02_code && python 41_tail_budget.py
"""
import os, json
import numpy as np
from collections import Counter
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
classes = json.load(open(BASE + "/data/classes_sorted.json"))
n2i = {c: i for i, c in enumerate(classes)}


def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n


def load(name):
    z = np.load(BASE + "/data/" + name, allow_pickle=True)
    lab = list(z["labels"])
    y = np.array([n2i[str(t)] for t in lab], dtype=np.int64) if isinstance(lab[0], (str, np.str_)) \
        else np.asarray(lab, dtype=np.int64)
    return norm(z["vectors"]), y


class CosNC:                     # 余弦最近质心
    def fit(self, X, y):
        self.cls = np.unique(y)
        self.W = np.stack([X[y == c].mean(0) for c in self.cls])
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12
        return self

    def predict(self, X):
        return self.cls[(X.astype(np.float32) @ self.W.T).argmax(1)]


def matched(X, y, k, seed, cohort):
    rng = np.random.RandomState(seed)
    idx = []
    for c in cohort:
        ci = np.where(y == c)[0]
        if len(ci) == 0:
            continue
        idx.append(ci if len(ci) <= k else rng.choice(ci, size=k, replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


# ---- cohort：基础集 < 30 条 且 真实测试集有句 ----
XB, yB = load("D_base_train.npz")
cntB = Counter(yB.tolist())
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XT_all, yT_all = norm(z2["vectors"][m]), np.asarray(z2["labels"], dtype=np.int64)[m]
cntT = Counter(yT_all.tolist())
cohort = sorted([c for c in range(679) if cntB.get(c, 0) < 30 and cntT.get(c, 0) >= 1])
mask = np.isin(yT_all, cohort)
Xtt, ytt = XT_all[mask], yT_all[mask]
print("cohort classes:", len(cohort), "| tail test rows:", len(ytt), flush=True)

XE, yE = load("D_final.npz")
XN, yN = load("D_naive_full.npz")

results = {}
for k in [10, 20, 30]:
    for jname in ["NC", "RIDGE"]:
        for name, (X, y) in [("BEDR", (XE, yE)), ("NAIVE", (XN, yN))]:
            accs = []
            for seed in [42, 43, 44]:
                Xs, ys = matched(X, y, k, seed, cohort)
                clf = CosNC().fit(Xs, ys) if jname == "NC" else RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys)
                accs.append(accuracy_score(ytt, clf.predict(Xtt)))
            results["k%d_%s_%s" % (k, jname, name)] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
            print("k=%d %-5s %-5s: %.4f +-%.4f" % (k, jname, name, np.mean(accs), np.std(accs)), flush=True)

json.dump({"cohort_classes": len(cohort), "tail_test_rows": int(len(ytt)),
           "cohort_definition": "base-set samples < 30 and >=1 real test sentence",
           "results": results},
          open(BASE + "/results/tail_budget_322.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
