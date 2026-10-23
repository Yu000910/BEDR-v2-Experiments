import numpy as np, json, os, time, csv
from collections import Counter
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

OUT = os.path.join(BASE, "out/round5")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
cohort = json.load(open(BASE + "/out/fix_pilot/tail_cohort.json"))["tail"]
TARGET = 30
CAP = 2.3

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zb = np.load(BASE + "/data/D_base_train.npz", allow_pickle=True)
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
with open(OUT + "/round5_candidates.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        cands.append({"tech": r["tech_id"], "sentence": r["sentence"], "plain_ent": float(r["plain_ent"]), "hybrid_ent": float(r["hybrid_ent"]), "maxsim": float(r["maxsim"])})
print("candidates:", len(cands), flush=True)
by_class = {}
for r in cands:
    by_class.setdefault(r["tech"], []).append(r)

def select(key, arm):
    chosen = {}
    for t, rs in by_class.items():
        c = name2idx[t]
        base = cnt.get(c, 0)
        need = max(0, TARGET - base)
        pool_rows = sorted(rs, key=lambda r: r[key])
        taken = []
        for r in pool_rows:
            if len(taken) >= need:
                break
            if r["maxsim"] >= 0.9:
                continue
            if r[key] < 0 or r[key] > CAP:
                continue
            taken.append(r)
        chosen[t] = taken
    return chosen

arms = {}
arms["NONE"] = select("plain_ent", "none")  # same select but with CAP high; handle below
# redo NONE with no entropy cap
def select_none():
    chosen = {}
    for t, rs in by_class.items():
        c = name2idx[t]
        base = cnt.get(c, 0)
        need = max(0, TARGET - base)
        taken = []
        for r in rs:
            if len(taken) >= need:
                break
            if r["maxsim"] >= 0.9:
                continue
            taken.append(r)
        chosen[t] = taken
    return chosen
arms["NONE"] = select_none()
arms["PLAIN"] = select("plain_ent", "plain")
arms["HYBRID"] = select("hybrid_ent", "hybrid")
for arm in arms:
    n = sum(len(v) for v in arms[arm].values())
    print(arm, "accepted:", n, "classes filled:", sum(1 for t, v in arms[arm].items() if cnt.get(name2idx[t], 0) + len(v) >= TARGET), "/", len(by_class), flush=True)

# embed unique accepted sentences
ec = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
unique = {}
for arm in arms:
    for t, taken in arms[arm].items():
        for r in taken:
            unique.setdefault(r["sentence"], None)
texts = list(unique.keys())
print("unique texts to embed:", len(texts), flush=True)
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
    if (i // B) % 200 == 0:
        print("embed", i, "/", len(texts), flush=True)
V = norm(np.vstack(vecs))
vmap = {t: v for t, v in zip(texts, V)}

for arm in arms:
    rows = []
    for t, taken in arms[arm].items():
        for r in taken:
            rows.append((name2idx[t], vmap[r["sentence"]], r["sentence"]))
    idx = [r[0] for r in rows]
    Xa = np.stack([r[1] for r in rows])
    Xout = np.vstack([XP, Xa]); yout = np.concatenate([yP, np.array(idx, dtype=np.int64)])
    texts_out = np.array(["pool" for _ in range(len(yP))] + [r[2] for r in rows])
    np.savez(OUT + "/D_round5_" + arm + ".npz", vectors=Xout, labels=yout, texts=texts_out)
    print("saved", arm, Xout.shape, "classes:", len(set(yout.tolist())), flush=True)
json.dump({"cohort": cohort}, open(OUT + "/arms_meta.json", "w"))
print("ARMS DONE", flush=True)

