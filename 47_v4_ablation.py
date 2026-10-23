# -*- coding: utf-8 -*-
"""text-embedding-v4 空间下的公平消融：结构化生成后【熵选】vs【顺序选】vs【随机选】
输出：全覆盖/真实语句准确率、难样本(LBE最高25%)准确率、Macro-F1，5种子配对检验。
"""
import os, csv, json, time
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from openai import OpenAI
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
KEY = open(BASE + "/../.dashscope_key").read().strip()
client = OpenAI(api_key=KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
CACHE = BASE + "/out/embed_cache_v4_candidates.npz"
classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

cands = []
for f in ["round5", "round6", "round6b", "round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if not os.path.exists(p): continue
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try: ms = float(r["maxsim"])
            except Exception: ms = 0.0
            cands.append({"order": len(cands), "tech": r["tech_id"].strip(), "s": r["sentence"].strip(), "ms": ms})
print("候选", len(cands), flush=True)

if os.path.exists(CACHE):
    VC = np.load(CACHE)["VC"]; print("用缓存候选嵌入", VC.shape, flush=True)
else:
    texts = [c["s"] for c in cands]; out = []; B = 10; t0 = time.time()
    for i in range(0, len(texts), B):
        for att in range(8):
            try:
                r = client.embeddings.create(model="text-embedding-v4", input=texts[i:i+B], dimensions=768, encoding_format="float")
                out.extend([np.array(e.embedding, dtype=np.float32) for e in r.data]); break
            except Exception as ex:
                if att == 7: raise
                time.sleep(2*att + 1)
        if (i // B) % 200 == 0: print("embed", i, "/", len(texts), round(time.time()-t0), "s", flush=True)
    VC = norm(np.vstack(out)); np.savez(CACHE, VC=VC); print("embed done", round(time.time()-t0), "s", flush=True)

zb = np.load(BASE + "/data/D_base_train.npz", allow_pickle=True)
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
print("候选混合空间 LBE: 均值 %.3f 中位 %.3f" % (h.mean(), np.median(h)), flush=True)

by = defaultdict(list)
for j, c in enumerate(cands):
    if c["tech"] in n2i: by[c["tech"]].append(j)
cntB = Counter(yB.tolist()); TARGET = 30
def select(mode, seed=42):
    rng = np.random.RandomState(seed); idx = []
    for t, js in by.items():
        c = n2i[t]; need = max(0, TARGET - cntB.get(c, 0))
        if need == 0: continue
        if mode == "ent": order = sorted(js, key=lambda j: h[j])
        elif mode == "order": order = sorted(js, key=lambda j: cands[j]["order"])
        else: order = list(rng.permutation(js))
        taken = 0
        for j in order:
            if taken >= need: break
            if cands[j]["ms"] >= 0.9: continue      # 去重（两臂相同）
            idx.append(j); taken += 1
    return np.array(sorted(set(idx)))
sel = {m: select(m) for m in ["ent", "order", "rand"]}
print("新增: " + " | ".join(f"{m} {len(v)}" for m, v in sel.items()), flush=True)
print("熵选与前序选重合 %.1f%%" % (100*len(np.intersect1d(sel["ent"], sel["order"]))/max(1,len(sel["ent"]))), flush=True)

def build(ai, seed=42):
    X = np.vstack([XB, VC[ai]]); y = np.concatenate([yB, yc[ai]])
    rng = np.random.RandomState(seed); keep = []
    for c in range(679):
        ci = np.where(y == c)[0]
        keep.append(ci if len(ci) <= 100 else rng.choice(ci, size=100, replace=False))
    k = np.sort(np.concatenate(keep)); return X[k], y[k]

zA = np.load(BASE + "/data/fullcov4291_plain.npz", allow_pickle=True); XA = norm(zA["vectors"]); yA = np.asarray(zA["labels"], dtype=np.int64)
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True); m2 = np.asarray(z2["src"]) == "real"
XR = norm(z2["vectors"][m2]); yR = np.asarray(z2["labels"], dtype=np.int64)[m2]
XRl = norm(XR + VI[yR]); S2 = XRl @ XBh.T; nr = np.argpartition(-S2, K, axis=1)[:, :K]; lr = yB[nr]
ht = np.zeros(len(yR), np.float32)
for j in range(len(lr)):
    v, c = np.unique(lr[j], return_counts=True); p = c / c.sum(); ht[j] = -(p * np.log2(p)).sum()
hard = ht >= np.percentile(ht, 75)
print("测试句 %d | 难样本 %d" % (len(yR), int(hard.sum())), flush=True)

res = {}
for mode in ["ent", "order", "rand"]:
    a679 = []; a420 = []; ah = []; mf = []
    for s in [42,43,44,45,46]:
        X, y = build(sel[mode], s)
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                            batch_size=256, learning_rate_init=1e-3, random_state=s).fit(X, y)
        pA = clf.predict(XA); pR = clf.predict(XR)
        a679.append(accuracy_score(yA, pA)); a420.append(accuracy_score(yR, pR))
        ah.append(accuracy_score(yR[hard], pR[hard]))
        mf.append(f1_score(yR, pR, average="macro", labels=np.unique(yR), zero_division=0))
    X, y = build(sel[mode], 42); rg = RidgeClassifier(alpha=1.0).fit(X, y)
    res[mode] = {"n": int(len(y)), "mlp679": [round(float(np.mean(a679)),4), round(float(np.std(a679)),4)],
                 "mlp420": [round(float(np.mean(a420)),4), round(float(np.std(a420)),4)],
                 "hard": [round(float(np.mean(ah)),4), round(float(np.std(ah)),4)],
                 "macroF1": round(float(np.mean(mf)),4),
                 "ridge679": round(float(accuracy_score(yA, rg.predict(XA))),4),
                 "ridge420": round(float(accuracy_score(yR, rg.predict(XR))),4),
                 "_raw_hard": [float(x) for x in ah]}
    print(mode, {k:v for k,v in res[mode].items() if not k.startswith("_")}, flush=True)

print("\n=== 配对检验（5 种子）===", flush=True)
for a, b in [("ent","order"), ("ent","rand")]:
    t, p = stats.ttest_rel(res[a]["_raw_hard"], res[b]["_raw_hard"])
    print(f"难样本 {a} vs {b}: Δ={100*(res[a]['hard'][0]-res[b]['hard'][0]):+.2f}pp t={t:.2f} p={p:.4f}", flush=True)
json.dump(res, open(BASE + "/results/ablation_v4_fair.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
