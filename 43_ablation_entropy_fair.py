# -*- coding: utf-8 -*-
"""公平消融：结构化生成 + 【有熵筛选】 vs 【无熵筛选】，除“选哪些候选”外完全一致。
用法：export DASHSCOPE_API_KEY=sk-xxx && python3 43_ablation_entropy_fair.py
"""
import os, csv, json, time, glob
import numpy as np
from collections import Counter
from openai import OpenAI
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("DASHSCOPE_API_KEY", "")
assert KEY, "请先设置环境变量 DASHSCOPE_API_KEY"
client = OpenAI(api_key=KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

def embed(texts, tag):
    out = []
    B = 10
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        for attempt in range(6):
            try:
                r = client.embeddings.create(model="text-embedding-v4", input=batch, dimensions=768, encoding_format="float")
                out.extend([np.array(e.embedding, dtype=np.float32) for e in r.data]); break
            except Exception as ex:
                if attempt == 5: raise
                time.sleep(2 * attempt + 1)
        if (i // B) % 100 == 0: print(tag, i, "/", len(texts), flush=True)
    return np.vstack(out)

# ---------- 1) 候选池（四轮，保持生成顺序） ----------
cands = []
for f in ["round5", "round6", "round6b", "round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if not os.path.exists(p): continue
    with open(p, newline="", encoding="utf-8") as fh:
        for j, r in enumerate(csv.DictReader(fh)):
            try: he = float(r["hybrid_ent"])
            except Exception: he = -1.0
            try: ms = float(r["maxsim"])
            except Exception: ms = 0.0
            cands.append({"round": f, "order": len(cands), "tech": r["tech_id"].strip(),
                          "sentence": r["sentence"].strip(), "hybrid_ent": he, "maxsim": ms})
print("候选总数:", len(cands), flush=True)

# ---------- 1b) 审计池：用【修复后】的熵重新接纳（两臂共用，保证对称） ----------
def audit_pool(theta=1.0):
    emb = np.load(BASE + "/data/audit/embeds.npz"); txts = json.load(open(BASE + "/data/audit/texts.json"))
    XC = np.vstack([norm(emb["rem_emb"]), norm(emb["kept_emb"])])
    yC = np.array([n2i[t] for t, _ in txts["removed"]] + [n2i[t] for t, _ in txts["kept"]], dtype=np.int64)
    return XC, yC, norm(np.vstack([zb["vectors"], XC]))

# ---------- 1c) 计算审计池（修复后的熵，θ=1.0，两臂共用） ----------
XC, yC, XPall = audit_pool()
K, TOP, BS = 15, 25, 2048
_h = np.zeros(len(XPall), np.float32)
for i in range(0, len(XPall), BS):
    D = 1.0 - XPall[i:i+BS] @ XPall.T
    _r = np.arange(D.shape[0]); D[_r, i+_r] = 1.0          # 仅排除自身（正确实现）
    part = np.argpartition(D, TOP-1, axis=1)[:, :TOP]
    labs = np.concatenate([yB, yC])[part[:, :K]]
    for j in range(len(labs)):
        v, c = np.unique(labs[j], return_counts=True)
        p = c / c.sum(); _h[i+j] = -(p * np.log2(p)).sum()
acc = _h[len(yB):] < 1.0
XAudit, yAudit = XC[acc], yC[acc]
print("审计池（修复后 θ=1.0）:", int(acc.sum()), "条", flush=True)

# ---------- 2) 基础集 ----------
zb = np.load(BASE + "/data/D_base_train.npz", allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([n2i[str(t)] for t in zb["labels"]], dtype=np.int64)
cntB = Counter(np.concatenate([yB, yC[acc]]).tolist())
TARGET = 30

by_class = {}
for r in cands: by_class.setdefault(r["tech"], []).append(r)

def select(mode):
    """mode='ent' 按混合熵升序；mode='order' 按生成顺序。其余条件完全相同。"""
    chosen = []
    for t, rs in by_class.items():
        if t not in n2i: continue
        c = n2i[t]; need = max(0, TARGET - cntB.get(c, 0))
        if need == 0: continue
        pool = sorted(rs, key=lambda r: r["hybrid_ent"]) if mode == "ent" else sorted(rs, key=lambda r: r["order"])
        taken = 0
        for r in pool:
            if taken >= need: break
            if r["maxsim"] >= 0.9: continue          # 去重规则（两臂相同）
            if r["hybrid_ent"] < 0: continue         # 缺意图向量的候选剔除（两臂相同）
            chosen.append(r); taken += 1
    return chosen

arms = {m: select(m) for m in ["ent", "order"]}
print({m: len(v) for m, v in arms.items()}, flush=True)

# ---------- 3) 只嵌入被选中的句子（取并集，避免重复调用） ----------
uniq = {}
for m, rows in arms.items():
    for r in rows: uniq.setdefault(r["sentence"], None)
texts = list(uniq.keys()); print("需嵌入的唯一句子:", len(texts), flush=True)
V = embed(texts, "embed")
for s, v in zip(texts, V): uniq[s] = v

# ---------- 4) 组装两臂 + 随机削顶到每类 100 ----------
XRv = norm(np.vstack([uniq[s] for s in texts]))
text_idx = {s: i for i, s in enumerate(texts)}
def build(mode, seed=42):
    rows = arms[mode]
    Xa = XRv[[text_idx[r["sentence"]] for r in rows]]
    ya = np.array([n2i[r["tech"]] for r in rows], dtype=np.int64)
    X = np.vstack([XB, XAudit, Xa]); y = np.concatenate([yB, yAudit, ya])
    # 随机削顶到每类 ≤100（两臂相同规则）
    rng = np.random.RandomState(seed); keep = []
    for c in range(679):
        ci = np.where(y == c)[0]
        keep.append(ci if len(ci) <= 100 else rng.choice(ci, size=100, replace=False))
    k = np.sort(np.concatenate(keep))
    return X[k], y[k]

# ---------- 5) 评估 ----------
zA = np.load(BASE + "/data/fullcov4291_plain.npz", allow_pickle=True)
XA_ = norm(zA["vectors"]); yA = np.asarray(zA["labels"], dtype=np.int64)
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True); msk = np.asarray(z2["src"]) == "real"
XR_ = norm(z2["vectors"][msk]); yR = np.asarray(z2["labels"], dtype=np.int64)[msk]

out = {}
for mode in ["ent", "order"]:
    X, y = build(mode)
    a, r = [], []
    for s in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                            batch_size=256, learning_rate_init=1e-3, random_state=s).fit(X, y)
        a.append(accuracy_score(yA, clf.predict(XA_))); r.append(accuracy_score(yR, clf.predict(XR_)))
    rg = RidgeClassifier(alpha=1.0).fit(X, y)
    out[mode] = {"n": int(len(y)), "mlp679": [round(float(np.mean(a)), 4), round(float(np.std(a)), 4)],
                 "mlp420": [round(float(np.mean(r)), 4), round(float(np.std(r)), 4)],
                 "ridge679": round(float(accuracy_score(yA, rg.predict(XA_))), 4),
                 "ridge420": round(float(accuracy_score(yR, rg.predict(XR_))), 4)}
    print(mode, out[mode], flush=True)
json.dump(out, open(BASE + "/results/ablation_entropy_fair.json", "w"), indent=1)
print("DONE ->", BASE + "/results/ablation_entropy_fair.json")
