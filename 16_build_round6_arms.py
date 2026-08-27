import numpy as np, json, os, time, csv
from collections import Counter
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

OUT = os.path.join(BASE, "out/round5")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
r6 = json.load(open(BASE + "/out/round5/r6_cohort.json"))["r6"]
TARGET = 30
CAP = 2.3

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([name2idx[str(t)] for t in zb["labels"]], dtype=np.int64)
emb = np.load(BASE + "/data/audit/embeds.npz")
txts = json.load(open(BASE + "/data/audit/texts.json"))
rem_lab = [name2idx[t] for t, _ in txts["removed"]]; kep_lab = [name2idx[t] for t, _ in txts["kept"]]
XC = np.vstack([norm(emb["rem_emb"]), norm(emb["kept_emb"])])
yC = np.array(rem_lab + kep_lab, dtype=np.int64)
fd = np.load(BASE + "/out/filter_corrected/filter_decision.npz")
XP = np.vstack([XB, XC[fd["accept_local10"]]]); yP = np.concatenate([yB, yC[fd["accept_local10"]]])
cnt = Counter(yP.tolist())

cands = []
for fn in ["round6_candidates.csv", "round6b_candidates.csv"]:
    with open(OUT + "/" + fn, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cands.append({"tech": r["tech_id"], "sentence": r["sentence"], "plain_ent": float(r["plain_ent"]), "hybrid_ent": float(r["hybrid_ent"]), "maxsim": float(r["maxsim"])})
print("r6 candidates:", len(cands), flush=True)
by_class = {}
for r in cands:
    by_class.setdefault(r["tech"], []).append(r)

def select(key, use_cap):
    chosen = {}
    for t, rs in by_class.items():
        c = name2idx[t]
        base = cnt.get(c, 0)
        need = max(0, TARGET - base)
        if use_cap:
            pool_rows = sorted(rs, key=lambda r: r[key])
        else:
            pool_rows = rs
        taken = []
        for r in pool_rows:
            if len(taken) >= need:
                break
            if r["maxsim"] >= 0.9:
                continue
            if use_cap and (r[key] < 0 or r[key] > CAP):
                continue
            taken.append(r)
        chosen[t] = taken
    return chosen

arms = {"NONE": select("plain_ent", False), "PLAIN": select("plain_ent", True), "HYBRID": select("hybrid_ent", True)}
for arm in arms:
    n = sum(len(v) for v in arms[arm].values())
    filled = sum(1 for t, v in arms[arm].items() if cnt.get(name2idx[t], 0) + len(v) >= TARGET)
    print(arm, "accepted:", n, "filled:", filled, "/", len(by_class), flush=True)

ec = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
unique = {}
for arm in arms:
    for t, taken in arms[arm].items():
        for r in taken:
            unique.setdefault(r["sentence"], None)
texts = list(unique.keys())
print("unique to embed:", len(texts), flush=True)
vecs = []
B = 10
for i in range(0, len(texts), B):
    batch = texts[i:i + B]
    for attempt in range(5):
        try:
            resp = ec.embeddings.create(model="text-embedding-v4", input=batch, dimensions=768, encoding_format="float")
            vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
            break
        except Exception as ex:
            if attempt == 4:
                print("FATAL", i, repr(ex)[:150], flush=True)
                raise
            time.sleep(2)
    if (i // B) % 100 == 0:
        print("embed", i, "/", len(texts), flush=True)
V = norm(np.vstack(vecs))
vmap = {t: v for t, v in zip(texts, V)}

def save_combined(arm):
    z5 = np.load(OUT + "/D_round5_" + arm + ".npz", allow_pickle=True)
    X5 = z5["vectors"].astype(np.float32); y5 = np.asarray(z5["labels"], dtype=np.int64)
    rows = []
    for t, taken in arms[arm].items():
        for r in taken:
            rows.append((name2idx[t], vmap[r["sentence"]]))
    if rows:
        Xa = np.stack([r[1] for r in rows]); ya = np.array([r[0] for r in rows], dtype=np.int64)
        Xout = np.vstack([X5, Xa]); yout = np.concatenate([y5, ya])
    else:
        Xout, yout = X5, y5
    np.savez(OUT + "/D_bedrv2_" + arm + ".npz", vectors=Xout, labels=yout)
    print("saved combined", arm, Xout.shape, "classes:", len(set(yout.tolist())), flush=True)
    return Xout, yout

# per-arm stats on zero-shot classes
zs = set(json.load(open(BASE + "/data/zero_shot_classes.json"))["zero_shot"])
zs_idx = [name2idx[c] for c in zs]
defn = {}
for l, v in zip(emb["def_labels"], emb["def_emb"]):
    defn[name2idx[str(l)]] = v

stats_out = {}
for arm in arms:
    Xc, yc = save_combined(arm)
    align = []
    for c in zs_idx:
        mask = yc == c
        if mask.sum() == 0:
            continue
        cent = Xc[mask].mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)
        dn = defn[c] / (np.linalg.norm(defn[c]) + 1e-12)
        align.append(float(cent @ dn))
    stats_out[arm] = {
        "zero_shot_covered": int(sum(1 for c in zs_idx if (yc == c).sum() > 0)),
        "zero_shot_alignment_mean": round(float(np.mean(align)), 4) if align else None,
    }
    print(arm, json.dumps(stats_out[arm]), flush=True)

json.dump(stats_out, open(OUT + "/round6_stats.json", "w"), indent=1)
print("R6 DONE", flush=True)

