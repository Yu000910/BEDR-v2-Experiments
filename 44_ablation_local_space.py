# -*- coding: utf-8 -*-
"""本地 SecureBERT 空间内的公平消融：结构化生成后【按熵选】vs【按生成顺序选】。
说明：因 DashScope 账号不可用，全部向量在本机用 SecureBERT 重新嵌入；绝对准确率与论文
（text-embedding-v4 空间）不可比，但“熵选 vs 顺序选”的差异是同一空间内的有效对照。"""
import os, csv, json, time
import numpy as np, torch
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained(BASE + "/../hf_models/SecureBERT")
mdl = AutoModel.from_pretrained(BASE + "/../hf_models/SecureBERT").eval()

@torch.no_grad()
def embed(texts, bs=64, tag=""):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=160, return_tensors="pt")
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
        e = torch.nn.functional.normalize(e, dim=1)
        out.append(e.numpy().astype(np.float32))
        if (i // bs) % 50 == 0: print(tag, i, "/", len(texts), flush=True)
    return np.vstack(out)

classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}
graph = json.load(open(BASE + "/data/intent_action_entity.json"))

# ---- 文本 ----
zb = np.load(BASE + "/data/D_base_train.npz", allow_pickle=True)
tB = [str(t) for t in zb["texts"]]; yB = np.array([n2i[str(x)] for x in zb["labels"]], dtype=np.int64)
cands = []
for f in ["round5", "round6", "round6b", "round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if not os.path.exists(p): continue
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try: he = float(r["hybrid_ent"])
            except Exception: he = -1.0
            try: ms = float(r["maxsim"])
            except Exception: ms = 0.0
            cands.append({"order": len(cands), "tech": r["tech_id"].strip(), "s": r["sentence"].strip(), "he": he, "ms": ms})
intent_texts = [str(graph.get(c, {}).get("intent", "") or classes[c]) for c in classes]
zr = np.load(BASE + "/../exp/bedr2/out/round5/test_real_texts.npz", allow_pickle=True)
tR = [str(t) for t in zr["texts"]]
print("base %d | cands %d | intents %d | test %d" % (len(tB), len(cands), len(intent_texts), len(tR)), flush=True)

t0 = time.time()
XB = embed(tB, tag="base")
VI = embed(intent_texts, tag="intent")
VC = embed([c["s"] for c in cands], tag="cand")
XR = embed(tR, tag="test")
print("embed done", round(time.time()-t0), "s", flush=True)

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"; yR = np.asarray(z2["labels"], dtype=np.int64)[m]
assert len(yR) == len(XR), (len(yR), len(XR))

# ---- 混合向量 ----
def hyb(X, y): return np.asarray([None]) if False else (lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True))(X + VI[y])
XBh = hyb(XB, yB)
VCh = (lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True))(VC + VI[np.array([n2i[c["tech"]] if c["tech"] in n2i else 0 for c in cands])])

# ---- 参考空间中的候选局部边界熵（k=15，池=基础集）----
K = 15
Sim = VCh @ XBh.T                      # 候选 × 基础集（余弦）
nb = np.argpartition(-Sim, K, axis=1)[:, :K]
labs = yB[nb]
h = np.zeros(len(cands), np.float32)
for j in range(len(labs)):
    v, c = np.unique(labs[j], return_counts=True); p = c / c.sum(); h[j] = -(p * np.log2(p)).sum()
print("候选混合空间 LBE: 均值 %.3f 中位 %.3f" % (h.mean(), np.median(h)), flush=True)

# ---- 两臂：只改“选哪些候选” ----
by = defaultdict(list)
for j, c in enumerate(cands):
    if c["tech"] in n2i: by[c["tech"]].append(j)
cntB = Counter(yB.tolist()); TARGET = 30
def select(mode):
    idx = []
    for t, js in by.items():
        c = n2i[t]; need = max(0, TARGET - cntB.get(c, 0))
        if need == 0: continue
        order = sorted(js, key=(lambda j: h[j]) if mode == "ent" else (lambda j: cands[j]["order"]))
        taken = 0
        for j in order:
            if taken >= need: break
            if cands[j]["he"] < 0: continue        # 无意图向量者剔除（两臂相同）
            idx.append(j); taken += 1
    return np.array(sorted(set(idx)))
ENT, ORD = select("ent"), select("order")
print("ENT 新增 %d 条 | ORD 新增 %d 条 | 新增样本的平均 LBE: ENT %.3f / ORD %.3f" % (len(ENT), len(ORD), h[ENT].mean(), h[ORD].mean()), flush=True)

def build(add_idx, seed=42):
    X = np.vstack([XB, VC[add_idx]]); y = np.concatenate([yB, np.array([n2i[cands[j]["tech"]] for j in add_idx])])
    rng = np.random.RandomState(seed); keep = []
    for c in range(679):
        ci = np.where(y == c)[0]
        keep.append(ci if len(ci) <= 100 else rng.choice(ci, size=100, replace=False))
    k = np.sort(np.concatenate(keep)); return X[k], y[k]

out = {}
for tag, ai in [("熵选", ENT), ("顺序选", ORD), ("全加（不筛选）", np.arange(len(cands)))]:
    X, y = build(ai); a = []
    for s in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                            batch_size=256, learning_rate_init=1e-3, random_state=s).fit(X, y)
        a.append(accuracy_score(yR, clf.predict(XR)))
    rg = RidgeClassifier(alpha=1.0).fit(X, y)
    out[tag] = {"n": int(len(y)), "mlp": [round(float(np.mean(a)), 4), round(float(np.std(a)), 4)],
                "ridge": round(float(accuracy_score(yR, rg.predict(XR))), 4),
                "added_mean_lbe": round(float(h[ai].mean()), 4) if len(ai) else None}
    print(tag, out[tag], flush=True)
json.dump(out, open(BASE + "/results/ablation_securebert_space.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
