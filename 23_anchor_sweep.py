import numpy as np, json
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5", exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([name2idx[str(t)] for t in zb["labels"]], dtype=np.int64)
emb = np.load(BASE + "/data/audit/embeds.npz")
txts = json.load(open(BASE + "/data/audit/texts.json"))
yC = np.array([name2idx[t] for t, _ in txts["removed"]] + [name2idx[t] for t, _ in txts["kept"]], dtype=np.int64)
XC = np.vstack([norm(emb["rem_emb"]), norm(emb["kept_emb"])])
fd = np.load(BASE + "/out/filter_corrected/filter_decision.npz")
XP = np.vstack([XB, XC[fd["accept_local10"]]]); yP = np.concatenate([yB, yC[fd["accept_local10"]]])

zint = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint["labels"], zint["vectors"])}
I = np.stack([intent_by_name[classes_sorted[c]] for c in yP])

def mean_local_entropy(V, k=15):
    h = np.zeros(len(V), dtype=np.float32)
    BS = 2048
    TOP = 16
    for i in range(0, len(V), BS):
        D = 1.0 - V[i:i+BS] @ V.T
        D[i:i+BS, i:i+BS] = 1.0
        part = np.argpartition(D, TOP-1, axis=1)[:, :TOP]
        idx = part[:, :k]
        labs = yP[idx]
        for r in range(len(idx)):
            vals, counts = np.unique(labs[r], return_counts=True)
            p = counts / counts.sum()
            h[i+r] = -(p * np.log2(p)).sum()
    return float(h.mean())

print("anchor strength | mean local boundary entropy")
out = {}
for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    V = norm((1.0 - alpha) * XP + alpha * I)
    e = mean_local_entropy(V)
    out[str(alpha)] = e
    print("  alpha=%.2f    %.4f" % (alpha, e))
json.dump(out, open(BASE + "/out/round5/anchor_sweep.json", "w"), indent=1)
print("DONE")

