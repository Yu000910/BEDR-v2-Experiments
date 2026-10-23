# -*- coding: utf-8 -*-
"""小样本预算下的熵筛选对照：只用生成的 k 条/类（k=5,10），比较熵选/顺序选/随机选。
评估面：真实语句测试集（全部/长尾）、全覆盖测试集（全部/长尾）。"""
import os, csv, json
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
BASE = os.path.dirname(os.path.abspath(__file__))
classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n
cands = []
for f in ["round5","round6","round6b","round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            try: ms = float(r["maxsim"])
            except Exception: ms = 0.0
            cands.append({"order": len(cands), "tech": r["tech_id"].strip(), "ms": ms})
VC = np.load(BASE + "/out/embed_cache_v4_candidates.npz")["VC"]
zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([n2i[str(x)] for x in zb["labels"]], dtype=np.int64)
zi = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
ib = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zi["labels"], zi["vectors"])}
VI = np.stack([ib[classes[c]] for c in range(679)])
yc = np.array([n2i[c["tech"]] if c["tech"] in n2i else 0 for c in cands])
XBh = norm(XB + VI[yB]); VCh = norm(VC + VI[yc])
K = 15; S = VCh @ XBh.T; nb = np.argpartition(-S, K, axis=1)[:, :K]; labs = yB[nb]
h = np.zeros(len(cands), np.float32)
for j in range(len(labs)):
    v, c = np.unique(labs[j], return_counts=True); p = c / c.sum(); h[j] = -(p * np.log2(p)).sum()
by = defaultdict(list)
for j, c in enumerate(cands):
    if c["tech"] in n2i: by[c["tech"]].append(j)
def select(mode, k, seed=42):
    rng = np.random.RandomState(seed); idx = []
    for t, js in by.items():
        if mode == "ent": order = sorted(js, key=lambda j: h[j])
        elif mode == "order": order = sorted(js, key=lambda j: cands[j]["order"])
        else: order = list(rng.permutation(js))
        taken = 0
        for j in order:
            if taken >= k: break
            if cands[j]["ms"] >= 0.9: continue
            idx.append(j); taken += 1
    return np.array(sorted(set(idx)))
# 评估面
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True); m2 = np.asarray(z2["src"]) == "real"
XR = norm(z2["vectors"][m2]); yR = np.asarray(z2["labels"], dtype=np.int64)[m2]
zA = np.load(BASE + "/data/orig4291_plain.npz", allow_pickle=True)
XA = norm(zA["vectors"]); yA = np.asarray(zA["labels"], dtype=np.int64)
cB = Counter(yB.tolist()); cT = Counter(yR.tolist())
tail = set(c for c in range(679) if cB.get(c, 0) < 30 and cT.get(c, 0) > 0)
faces = {"真实语句·全部": (XR, yR, np.ones(len(yR), bool)),
         "真实语句·长尾": (XR, yR, np.isin(yR, list(tail))),
         "全覆盖·全部": (XA, yA, np.ones(len(yA), bool)),
         "全覆盖·长尾": (XA, yA, np.isin(yA, list(tail)))}
print("长尾类 %d 个；各评估面行数：" % len(tail), {k: int(v[2].sum()) for k, v in faces.items()}, flush=True)
res = {}
for k in [5, 10]:
    for mode in ["ent", "order", "rand"]:
        accs = {f: [] for f in faces}
        for s in [42, 43, 44, 45, 46]:
            ai = select(mode, k)
            X, y = VC[ai], yc[ai]
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                                batch_size=256, learning_rate_init=1e-3, random_state=s).fit(X, y)
            for f, (XX, yy, mk) in faces.items():
                accs[f].append(accuracy_score(yy[mk], clf.predict(XX[mk])))
        res[f"k{k}_{mode}"] = {f: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for f, v in accs.items()}
        res[f"k{k}_{mode}"]["_raw"] = {f: [float(x) for x in v] for f, v in accs.items()}
        print(f"k={k} {mode:5s}", {f: res[f'k{k}_{mode}'][f][0] for f in faces}, flush=True)
print("\n=== 配对检验（5 种子）===", flush=True)
tests = {}
for k in [5, 10]:
    for a, b, nm in [("ent", "order", "熵选vs顺序"), ("ent", "rand", "熵选vs随机")]:
        for f in faces:
            t, p = stats.ttest_rel(res[f"k{k}_{a}"]["_raw"][f], res[f"k{k}_{b}"]["_raw"][f])
            dd = 100 * (np.mean(res[f"k{k}_{a}"]["_raw"][f]) - np.mean(res[f"k{k}_{b}"]["_raw"][f]))
            tests[f"k{k}|{f}|{nm}"] = [round(dd, 2), round(float(t), 2), round(float(p), 4)]
            print(f"k={k} {f:12s} {nm}: Δ={dd:+.2f}pp t={t:+.2f} p={p:.4f}", flush=True)
json.dump({"results": res, "tests": tests}, open(BASE + "/results/entropy_smallbudget.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
