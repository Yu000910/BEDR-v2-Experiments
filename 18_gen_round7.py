import numpy as np, json, os, time, csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

PROJ = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out/round5")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
graph = json.load(open(os.path.join(BASE, "data/intent_action_entity.json")))
TARGET = 30
K = 15
CALLS = 5

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

z = np.load(OUT + "/D_bedr_HYBRID.npz", allow_pickle=True)
XP = norm(z["vectors"].astype(np.float32)); yP = np.asarray(z["labels"], dtype=np.int64)
cnt = Counter(yP.tolist())
under = sorted([c for c in range(679) if cnt.get(c, 0) < TARGET])
print("under-30 classes:", len(under), [classes_sorted[c] for c in under], flush=True)

zint = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint["labels"], zint["vectors"])}
XH = norm(XP + np.stack([intent_by_name[classes_sorted[c]] for c in yP]))
print("pools:", XP.shape, XH.shape, flush=True)

llm = OpenAI(api_key="YOUR_DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")
ec = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def embed_batch(texts):
    out = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        for attempt in range(5):
            try:
                resp = ec.embeddings.create(model="text-embedding-v4", input=batch, dimensions=768, encoding_format="float")
                out.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
                break
            except Exception as ex:
                if attempt == 4:
                    print("FATAL embed", repr(ex)[:150], flush=True)
                    raise
                time.sleep(2)
    return np.stack(out)

def entropy_of(D, labs, k=K):
    top = np.argpartition(D, k - 1)[:k]
    vals, counts = np.unique(labs[top], return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def process_class(c):
    t = classes_sorted[c]
    g = graph.get(t, {})
    intent = g.get("intent", "")
    actions = chr(10).join(["- " + d.get("verb", "") + ": " + d.get("tool", "") for d in g.get("action_chain", [])])
    prompt = ("As a cybersecurity expert, your task is to generate 15 diverse, realistic, and concise example sentences for the ATT&CK technique " + t + "." + chr(10)
        + "**Key Objective (Intent):**" + chr(10) + intent + chr(10)
        + "**Generate sentences that clearly describe one of these core actions:**" + chr(10) + actions + chr(10)
        + "**CRITICAL INSTRUCTIONS:**" + chr(10)
        + "1. LANGUAGE: English only." + chr(10)
        + "2. Each sentence must be a clear example of ONE specific action, not a long story." + chr(10)
        + "3. Vary the actions and tools across sentences; do not repeat the same sentence with minor changes." + chr(10)
        + "4. OUTPUT ONLY a valid JSON list of exactly 15 strings." + chr(10)
        + "**Example of a good, focused sentence:**" + chr(10)
        + "The malware used 'vssadmin delete shadows' to erase volume shadow copies and hinder system recovery.")
    intent_vec = intent_by_name.get(t)
    rows = []
    same_idx = np.where(yP == c)[0]
    for call_i in range(CALLS):
        sents = None
        for attempt in range(4):
            try:
                rsp = llm.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = rsp.choices[0].message.content
                sents = json.loads(content)
                if isinstance(sents, dict):
                    sents = next((v for v in sents.values() if isinstance(v, list)), [])
                if isinstance(sents, list) and len(sents) >= 10:
                    break
                sents = None
            except Exception as ex:
                print(t, "llm attempt", attempt, repr(ex)[:100], flush=True)
                time.sleep(2)
        if not sents:
            continue
        sents = [str(s).strip() for s in sents[:15] if isinstance(s, str) and len(str(s)) >= 12]
        if not sents:
            continue
        try:
            vecs = norm(embed_batch(sents))
        except Exception:
            continue
        for s, v in zip(sents, vecs):
            maxsim = float((v @ XP[same_idx].T).max()) if len(same_idx) else 0.0
            plain_ent = entropy_of(1.0 - v @ XP.T, yP)
            if intent_vec is not None:
                hv = norm((v + intent_vec).reshape(1, -1))[0]
                hybrid_ent = entropy_of(1.0 - hv @ XH.T, yP)
            else:
                hybrid_ent = -1.0
            rows.append({"tech_id": t, "sentence": s, "plain_ent": round(plain_ent, 4), "hybrid_ent": round(hybrid_ent, 4), "maxsim": round(maxsim, 4)})
    return c, rows

all_rows = []
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(process_class, c): c for c in under}
    done = 0
    for fut in as_completed(futs):
        c, rows = fut.result()
        all_rows.extend(rows)
        done += 1
        if done % 12 == 0:
            print("progress", done, "/", len(under), "candidates", len(all_rows), flush=True)

with open(OUT + "/round7_candidates.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=["tech_id", "sentence", "plain_ent", "hybrid_ent", "maxsim"])
    wr.writeheader()
    wr.writerows(all_rows)
print("round7 candidates saved:", len(all_rows), flush=True)

# ---------- fill ----------
cands = []
with open(OUT + "/round7_candidates.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        cands.append({"tech": r["tech_id"], "sentence": r["sentence"], "hybrid_ent": float(r["hybrid_ent"]), "maxsim": float(r["maxsim"])})
by_class = {}
for r in cands:
    by_class.setdefault(r["tech"], []).append(r)
add_texts = []
add_idx = []
for c in under:
    t = classes_sorted[c]
    rs = sorted(by_class.get(t, []), key=lambda r: r["hybrid_ent"])
    n = cnt.get(c, 0)
    for r in rs:
        if n >= TARGET:
            break
        if r["maxsim"] >= 0.92:
            continue
        add_texts.append(r["sentence"]); add_idx.append(c); n += 1
print("to add:", len(add_texts), flush=True)
if add_texts:
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
            print("fill embed", i, "/", len(add_texts), flush=True)
    Va = norm(np.vstack(vecs))
    Xout = np.vstack([XP, Va]); yout = np.concatenate([yP, np.array(add_idx, dtype=np.int64)])
    np.savez(OUT + "/D_bedr_HYBRID.npz", vectors=Xout, labels=yout)
    cnt2 = Counter(yout.tolist())
    vals = sorted(cnt2.values())
    print("HYBRID refilled rows", len(yout), "classes", len(cnt2), "min", vals[0], "median", vals[len(vals)//2], "max", vals[-1], "under30:", sum(1 for v in vals if v < 30), flush=True)
print("R7 DONE", flush=True)

