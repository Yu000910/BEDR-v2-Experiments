import numpy as np, json, csv, time
from collections import Counter
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE + "/out/round5"
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
TARGET = 30

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

cands = []
for fn in ["round6_candidates.csv", "round6b_candidates.csv"]:
    with open(OUT + "/" + fn, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cands.append({"tech": r["tech_id"], "sentence": r["sentence"], "plain_ent": float(r["plain_ent"]), "hybrid_ent": float(r["hybrid_ent"]), "maxsim": float(r["maxsim"])})
by_class = {}
for r in cands:
    by_class.setdefault(r["tech"], []).append(r)
print("candidates:", len(cands), "classes:", len(by_class), flush=True)

def fill(arm):
    key = "plain_ent" if arm == "PLAIN" else "hybrid_ent"
    z = np.load(OUT + "/D_bedrv2_" + arm + ".npz", allow_pickle=True)
    X = z["vectors"].astype(np.float32); y = np.asarray(z["labels"], dtype=np.int64)
    cnt = Counter(y.tolist())
    under = [c for c in range(679) if cnt.get(c, 0) < TARGET]
    print(arm, "under-filled classes:", len(under), flush=True)
    add_texts = []
    add_idx = []
    for c in under:
        t = classes_sorted[c]
        rs = sorted(by_class.get(t, []), key=lambda r: r[key])
        n = cnt.get(c, 0)
        for r in rs:
            if n >= TARGET:
                break
            if r["maxsim"] >= 0.9:
                continue
            add_texts.append(r["sentence"]); add_idx.append(c); n += 1
    if not add_texts:
        print(arm, "nothing to add", flush=True)
        return
    ec = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    vecs = []
    B = 10
    for i in range(0, len(add_texts), B):
        batch = add_texts[i:i + B]
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
        if (i // B) % 50 == 0:
            print(arm, "embed", i, "/", len(add_texts), flush=True)
    Va = norm(np.vstack(vecs))
    Xout = np.vstack([X, Va]); yout = np.concatenate([y, np.array(add_idx, dtype=np.int64)])
    np.savez(OUT + "/D_bedrv2_" + arm + ".npz", vectors=Xout, labels=yout)
    cnt2 = Counter(yout.tolist())
    vals = sorted(cnt2.values())
    print(arm, "final rows", len(yout), "classes", len(cnt2), "min", vals[0], "median", vals[len(vals)//2], "max", vals[-1], flush=True)

fill("PLAIN")
fill("HYBRID")

# zero-shot alignment check after fill
zs = set(json.load(open(BASE + "/data/zero_shot_classes.json"))["zero_shot"])
zs_idx = [name2idx[c] for c in zs]
emb = np.load(BASE + "/data/audit/embeds.npz")
defn = {}
for l, v in zip(emb["def_labels"], emb["def_emb"]):
    defn[name2idx[str(l)]] = v
for arm in ["NONE", "PLAIN", "HYBRID"]:
    z = np.load(OUT + "/D_bedrv2_" + arm + ".npz", allow_pickle=True)
    X = norm(z["vectors"].astype(np.float32)); y = np.asarray(z["labels"], dtype=np.int64)
    aligns = []
    for c in zs_idx:
        m = y == c
        if m.sum() == 0:
            continue
        cent = X[m].mean(axis=0); cent = cent / (np.linalg.norm(cent) + 1e-12)
        dn = defn[c] / (np.linalg.norm(defn[c]) + 1e-12)
        aligns.append(float(cent @ dn))
    print(arm, "zero-shot covered:", len(aligns), "alignment:", round(float(np.mean(aligns)), 4), flush=True)
print("FILL DONE", flush=True)

