import numpy as np, json, os, time, csv
from collections import Counter
from openai import OpenAI
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/fix_pilot/", exist_ok=True)

PROJ = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out/fix_pilot")
os.makedirs(OUT, exist_ok=True)
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
graph = json.load(open(os.path.join(BASE, "data/intent_action_entity.json")))

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
yT = np.asarray(z2["labels"], dtype=np.int64)[m]
cnt = Counter(yT.tolist())
top20 = [c for c, _ in cnt.most_common(20)]
top20_names = [classes_sorted[c] for c in top20]
print("pilot classes:", top20_names, flush=True)
lens = [len(graph.get(t, {}).get("action_chain", [])) for t in top20_names]
print("action_chain lengths:", lens, "max", max(lens), flush=True)

# ---------- 1. generation (fix) ----------
csv_path = os.path.join(OUT, "fix_pilot.csv")
if os.path.exists(csv_path):
    print("csv exists, skip generation", flush=True)
else:
    client = OpenAI(api_key="YOUR_DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")
    all_rows = []
    for t in top20_names:
        g = graph.get(t, {})
        intent = g.get("intent", "")
        actions = chr(10).join(["- " + d.get("verb", "") + " via " + d.get("tool", "") for d in g.get("action_chain", [])])
        entities = ", ".join(g.get("entities", []))
        prompt = "As a cybersecurity expert, generate 15 diverse example sentences for the ATT&CK technique " + t + ".\nKey objective (intent): " + intent + "\nCore actions (use these but vary tools and phrasings):\n" + actions + "\nPlausible entities: " + entities + "\nCRITICAL INSTRUCTIONS:\n1. ENGLISH only.\n2. DIVERSITY: every sentence must be clearly different from every other - vary the actor (different malware names, scripts, users, threat actors), the tool, the sentence structure and the wording. Never output two sentences that say the same thing with minor word changes. Avoid near-duplicates.\n3. Each sentence describes ONE specific action and is concise.\n4. Return ONLY a valid JSON list of exactly 15 strings."
        ok = False
        for attempt in range(5):
            try:
                rsp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = rsp.choices[0].message.content
                sents = json.loads(content)
                if isinstance(sents, dict):
                    sents = next((v for v in sents.values() if isinstance(v, list)), [])
                if isinstance(sents, list) and len(sents) >= 10:
                    all_rows.extend([{"tech_id": t, "sentence": str(s).strip()} for s in sents[:15]])
                    ok = True
                    break
                print(t, "bad response", repr(content)[:100], flush=True)
            except Exception as ex:
                print(t, "attempt", attempt, repr(ex)[:150], flush=True)
                time.sleep(2)
        if not ok:
            print("FAILED class", t, flush=True)
        print(t, "done, total rows", len(all_rows), flush=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["tech_id", "sentence"])
        wr.writeheader()
        wr.writerows(all_rows)
    print("saved", len(all_rows), "rows", flush=True)

# ---------- 2. embedding ----------
npz_path = os.path.join(OUT, "fix_pilot_plain.npz")
if os.path.exists(npz_path):
    print("npz exists, skip embedding", flush=True)
else:
    ec = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["tech_id"].strip(), r["sentence"].strip()))
    texts = [s for _, s in rows]
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
                    print("FATAL embed", i, repr(ex)[:200], flush=True)
                    raise
                time.sleep(2)
        if (i // B) % 20 == 0:
            print("embed", i, "/", len(texts), flush=True)
    V = np.vstack(vecs).astype(np.float32)
    np.savez(npz_path, vectors=V, labels=np.array([t for t, _ in rows]), texts=np.array(texts))
    print("embedded", V.shape, flush=True)

# ---------- 3. eval ----------
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

zfix = np.load(npz_path, allow_pickle=True)
Xfix = norm(zfix["vectors"]); yfix = np.array([name2idx[str(t)] for t in zfix["labels"]], dtype=np.int64)
zn = np.load(BASE + "/data/D_naive_full.npz", allow_pickle=True)
Xnai = norm(zn["vectors"]); ynai = np.array([name2idx[str(t)] for t in zn["labels"]], dtype=np.int64)
zold = np.load(BASE + "/data/D_bedr_train.npz", allow_pickle=True)
Xold = norm(zold["vectors"]); yold = np.array([name2idx[str(t)] for t in zold["labels"]], dtype=np.int64)
XT = norm(z2["vectors"][m])

emb = np.load(BASE + "/data/audit/embeds.npz")
defn = {str(l): v for l, v in zip(emb["def_labels"], emb["def_emb"])}
defn = {name2idx[k]: v for k, v in defn.items()}

tmask = np.isin(yT, top20)
Xte, yte = XT[tmask], yT[tmask]
print("eval test rows:", Xte.shape[0], "classes:", len(set(yte.tolist())), flush=True)

def sample30(X, y, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for c in top20:
        ci = np.where(y == c)[0]
        if len(ci) >= 30:
            parts.append(rng.choice(ci, size=30, replace=False))
        else:
            parts.append(ci)
            print("   class", classes_sorted[c], "only", len(ci), "samples", flush=True)
    return X[np.concatenate(parts)], y[np.concatenate(parts)]

class CosNC20:
    def fit(self, X, y):
        self.classes_ = top20
        C = np.stack([X[y == c].mean(axis=0) for c in top20])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.W_ = C.astype(np.float32)
        return self
    def predict(self, X):
        idx = (X.astype(np.float32) @ self.W_.T).argmax(axis=1)
        return np.array([self.classes_[i] for i in idx])

def stats(X, y):
    rec = {}
    for c in top20:
        Xc = X[y == c]
        S = Xc @ Xc.T
        ii, jj = np.triu_indices(len(Xc), k=1)
        rec[c] = {"mean_pair": float(S[ii, jj].mean()), "max_pair": float(S[ii, jj].max())}
        cent = Xc.mean(axis=0)
        cn = cent / (np.linalg.norm(cent) + 1e-12)
        mm = yte == c
        if mm.sum() > 0:
            rec[c]["test_cov"] = float((Xte[mm] @ cn).mean())
        rec[c]["fidelity"] = float((Xc @ (defn[c] / (np.linalg.norm(defn[c]) + 1e-12))).mean())
    return rec

rows = {}
datasets = {"FIX": (Xfix, yfix), "NAIVE": (Xnai, ynai), "OLD": (Xold, yold)}
for name, (X, y) in datasets.items():
    acc_nc = []; acc_ridge = []
    for seed in [42, 43, 44]:
        Xs, ys = sample30(X, y, seed)
        acc_nc.append(accuracy_score(yte, CosNC20().fit(Xs, ys).predict(Xte)))
        acc_ridge.append(accuracy_score(yte, RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys).predict(Xte)))
    st = stats(Xs if "Xs" in dir() else X, ys if "Ys" in dir() else y)
    rows[name] = {
        "NC_acc": [round(float(np.mean(acc_nc)), 4), round(float(np.std(acc_nc)), 4)],
        "Ridge_acc": [round(float(np.mean(acc_ridge)), 4), round(float(np.std(acc_ridge)), 4)],
        "fidelity": round(float(np.mean([v["fidelity"] for v in st.values()])), 4),
        "mean_pair": round(float(np.mean([v["mean_pair"] for v in st.values()])), 4),
        "max_pair": round(float(np.mean([v["max_pair"] for v in st.values()])), 4),
        "test_cov": round(float(np.mean([v["test_cov"] for v in st.values() if "test_cov" in v])), 4),
    }
    print(name, json.dumps(rows[name]), flush=True)
json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("DONE", flush=True)

