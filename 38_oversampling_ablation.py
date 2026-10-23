# -*- coding: utf-8 -*-
"""38_oversampling_ablation.py
多轮自适应过采样消融（对应第三章表3.14 第5行）：
  基础集（D_base_train_new）→ 多轮自适应过采样后（D_arms_HYBRID）→ 终版（D_final，含熵引导选择性欠采样）
指标：真实语句测试集（2,984 条）与全覆盖测试集（679 类、4,291 条）上的 MLP（5 种子）/ Ridge 准确率。
输出：results/oversampling_ablation.json
"""
import numpy as np, json, os
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                         batch_size=256, learning_rate_init=1e-3, random_state=seed)


arms = [("baseline", D + "/D_base_train_new.npz", "name"),
        ("oversampled", D + "/D_arms_HYBRID.npz", "int"),
        ("final", D + "/D_final.npz", "int")]

z2 = np.load(D + "/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
XBt = norm(z2["vectors"][m].astype(np.float32)); yBt = np.asarray(z2["labels"], dtype=np.int64)[m]
zA = np.load(D + "/orig4291_plain.npz", allow_pickle=True)
XAt = norm(zA["vectors"].astype(np.float32)); yAt = np.asarray(zA["labels"], dtype=np.int64)
print("real test:", XBt.shape[0], "| full-679 test:", XAt.shape[0], flush=True)

out = {}
for name, p, kind in arms:
    X, y = load(p, kind)
    row = {"n": int(len(y)), "classes": int(len(set(y.tolist())))}
    for jname in ["MLP", "RIDGE"]:
        real, full = [], []
        for seed in [42, 43, 44, 45, 46]:
            clf = (mlp(seed) if jname == "MLP" else RidgeClassifier(alpha=1.0, random_state=seed)).fit(X, y)
            real.append(accuracy_score(yBt, clf.predict(XBt)))
            full.append(accuracy_score(yAt, clf.predict(XAt)))
        row["real_" + jname] = [round(float(np.mean(real)), 4), round(float(np.std(real)), 4)]
        row["full679_" + jname] = [round(float(np.mean(full)), 4), round(float(np.std(full)), 4)]
    out[name] = row
    print(name, json.dumps(row, ensure_ascii=False), flush=True)

json.dump(out, open(BASE + "/results/oversampling_ablation.json", "w"), indent=1, ensure_ascii=False)
print("DONE ->", BASE + "/results/oversampling_ablation.json", flush=True)
