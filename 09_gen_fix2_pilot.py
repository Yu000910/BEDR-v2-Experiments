import numpy as np, json, os, time, csv
from collections import Counter
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/fix_pilot/", exist_ok=True)

PROJ = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out/fix_pilot")
classes_sorted = json.load(open(BASE + "/data/classes_sorted.json"))
name2idx = {c: i for i, c in enumerate(classes_sorted)}
graph = json.load(open(os.path.join(BASE, "data/intent_action_entity.json")))

z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"
yT = np.asarray(z2["labels"], dtype=np.int64)[m]
cnt = Counter(yT.tolist())
top20 = [c for c, _ in cnt.most_common(20)]
top20_names = [classes_sorted[c] for c in top20]

zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
base_examples = {}
for l, t in zip(zb["labels"], zb["texts"]):
    base_examples.setdefault(str(l), []).append(str(t))

csv_path = os.path.join(OUT, "fix2_pilot.csv")
if os.path.exists(csv_path):
    print("csv exists, skip generation", flush=True)
else:
    client = OpenAI(api_key="YOUR_DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")
    all_rows = []
    for t in top20_names:
        g = graph.get(t, {})
        intent = g.get("intent", "")
        actions = chr(10).join(["- " + d.get("verb", "") + " via " + d.get("tool", "") for d in g.get("action_chain", [])])
        exs = base_examples.get(t, [])[:3]
        ex_str = chr(10).join([str(i + 1) + ". " + e for i, e in enumerate(exs)])
        prompt = "As a cybersecurity expert, generate 15 new example sentences for the ATT&CK technique " + t + ".\nTechnique intent: " + intent + "\nCore actions:\n" + actions + "\nStyle examples (from real threat reports):\n" + ex_str + "\nWrite 15 NEW sentences in the exact same style: short, factual, each one attributed to a DIFFERENT plausible malware or tool name (vary the names across sentences). Do not copy the examples. Do not repeat the same sentence with minor changes.\nReturn ONLY a valid JSON list of exactly 15 strings."
        for call_i in range(2):
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
                    print(t, "call", call_i, "bad", repr(content)[:80], flush=True)
                except Exception as ex:
                    print(t, "call", call_i, "attempt", attempt, repr(ex)[:120], flush=True)
                    time.sleep(2)
            if not ok:
                print("FAILED class", t, "call", call_i, flush=True)
        print(t, "done, total", len(all_rows), flush=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["tech_id", "sentence"])
        wr.writeheader()
        wr.writerows(all_rows)
    print("saved", len(all_rows), flush=True)

npz_path = os.path.join(OUT, "fix2_pilot_plain.npz")
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
        if (i // B) % 30 == 0:
            print("embed", i, "/", len(texts), flush=True)
    V = np.vstack(vecs).astype(np.float32)
    np.savez(npz_path, vectors=V, labels=np.array([t for t, _ in rows]), texts=np.array(texts))
    print("embedded", V.shape, flush=True)
print("GEN DONE", flush=True)

