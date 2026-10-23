import numpy as np, json, os, time, csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

PROJ = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out/round5")
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
graph = json.load(open(os.path.join(BASE, "data/intent_action_entity.json")))
cohort = json.load(open(BASE + "/out/round5/r6_cohort.json"))["r6"]
TARGET = 30
K = 15
CALLS_PER_CLASS = 3

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
acc10 = fd["accept_local10"]
XP = np.vstack([XB, XC[acc10]]); yP = np.concatenate([yB, yC[acc10]])
print("plain pool:", XP.shape, flush=True)

zint = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint["labels"], zint["vectors"])}
missing_intent = [c for c in set(yP.tolist()) if classes_sorted[c] not in intent_by_name]
print("pool classes missing intent:", len(missing_intent), flush=True)
XH = norm(XP + np.stack([intent_by_name[classes_sorted[c]] for c in yP]))
print("hybrid pool:", XH.shape, flush=True)

cnt = Counter(yP.tolist())
cohort = [c for c in cohort if cnt.get(c, 0) < TARGET]
print("cohort:", len(cohort), flush=True)

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
    for call_i in range(CALLS_PER_CLASS):
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
    futs = {ex.submit(process_class, c): c for c in cohort}
    done = 0
    for fut in as_completed(futs):
        c, rows = fut.result()
        all_rows.extend(rows)
        done += 1
        if done % 40 == 0:
            print("progress", done, "/", len(cohort), "candidates", len(all_rows), flush=True)

with open(OUT + "/round6_candidates.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=["tech_id", "sentence", "plain_ent", "hybrid_ent", "maxsim"])
    wr.writeheader()
    wr.writerows(all_rows)
print("candidates saved:", len(all_rows), flush=True)
pe = [r["plain_ent"] for r in all_rows if r["plain_ent"] > 0]
he = [r["hybrid_ent"] for r in all_rows if r["hybrid_ent"] > 0]
print("plain_ent mean %.3f median %.3f | hybrid_ent mean %.3f median %.3f" % (np.mean(pe), np.median(pe), np.mean(he), np.median(he)), flush=True)
print("DONE", flush=True)

