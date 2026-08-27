import json, os, time, sys, csv
import numpy as np
from openai import OpenAI
from collections import defaultdict
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/fidelity_audit/", exist_ok=True)

OUT = os.path.join(BASE, "out/fidelity_audit")
os.makedirs(OUT, exist_ok=True)
client = OpenAI(api_key="YOUR_DASHSCOPE_API_KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def embed(texts, tag):
    vecs = []
    B = 10
    t0 = time.time()
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        ok = False
        for attempt in range(6):
            try:
                resp = client.embeddings.create(model="text-embedding-v4", input=batch, dimensions=768, encoding_format="float")
                vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
                ok = True
                break
            except Exception as ex:
                if attempt == 5:
                    print("FATAL", tag, i, repr(ex)[:200], flush=True)
                    sys.exit(1)
                time.sleep(2 * attempt + 1)
        if (i // B) % 100 == 0:
            print(tag, i, "/", len(texts), round(time.time() - t0), "s", flush=True)
    return np.vstack(vecs).astype(np.float32)

d = json.load(open(os.path.join(BASE, "data/enterprise-attack.json")))
objs = d.get("objects", d)
defs = {}
for o in objs:
    if o.get("type") != "attack-pattern":
        continue
    eid = None
    for er in o.get("external_references", []):
        if er.get("source_name") == "mitre-attack":
            eid = er.get("external_id")
    if eid:
        defs[eid] = (str(o.get("name", "")) + ". " + str(o.get("description") or "")).strip()
classes_sorted = json.load(open(os.path.join(BASE, "data/classes_sorted.json")))
missing = [c for c in classes_sorted if c not in defs]
print("definitions:", len(defs), "missing of 679:", len(missing), missing[:10], flush=True)
def_labels = [c for c in classes_sorted if c in defs]
def_texts = [defs[c] for c in def_labels]

def read_csv(f):
    rows = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["tech_id"].strip(), r["sentence"].strip()))
    return rows

proj = os.path.join(BASE, "data")
r2 = read_csv(proj + "aug_sentences_second_pass.csv")
r3 = read_csv(proj + "aug_sentences_third_pass.csv")
r4 = read_csv(proj + "aug_sentences_fourth_pass.csv")
kept = read_csv(proj + "aug_sentences_final_filtered.csv")
union234 = set(r2) | set(r3) | set(r4)
kept_set = set(kept)
removed = sorted(union234 - kept_set)
kept_list = sorted(kept_set)
naive_rows = read_csv(proj + "aug_sentences_naive_llm_direct.csv")
rng = np.random.RandomState(42)
naive_samp = [naive_rows[i] for i in rng.choice(len(naive_rows), size=min(5000, len(naive_rows)), replace=False)]
print("populations: removed", len(removed), "kept", len(kept_list), "naive_sample", len(naive_samp), flush=True)

def_emb = embed(def_texts, "DEF")
rem_emb = embed([t for _, t in removed], "REM")
kept_emb = embed([t for _, t in kept_list], "KEPT")
naive_emb = embed([t for _, t in naive_samp], "NAIVE")
np.savez(os.path.join(OUT, "embeds.npz"), def_emb=def_emb, def_labels=np.array(def_labels), rem_emb=rem_emb, kept_emb=kept_emb, naive_emb=naive_emb)
json.dump({"removed": removed, "kept": kept_list, "naive": naive_samp}, open(os.path.join(OUT, "texts.json"), "w"))

defn = {l: v for l, v in zip(def_labels, def_emb)}
def fidelity(emb, labels):
    out = []
    for e, l in zip(emb, labels):
        if l in defn:
            dn = defn[l] / (np.linalg.norm(defn[l]) + 1e-12)
            out.append(float((e / (np.linalg.norm(e) + 1e-12)) @ dn))
        else:
            out.append(None)
    return out

f_rem = [x for x in fidelity(rem_emb, [l for l, _ in removed]) if x is not None]
f_kep = [x for x in fidelity(kept_emb, [l for l, _ in kept_list]) if x is not None]
f_nai = [x for x in fidelity(naive_emb, [l for l, _ in naive_samp]) if x is not None]
print("overall fidelity means: removed %.4f (n=%d), kept %.4f (n=%d), naive %.4f (n=%d)" % (np.mean(f_rem), len(f_rem), np.mean(f_kep), len(f_kep), np.mean(f_nai), len(f_nai)), flush=True)

def within_class(emb, labels, drop_missing=True):
    by = defaultdict(list)
    for e, l in zip(emb, labels):
        if l in defn:
            dn = defn[l] / (np.linalg.norm(defn[l]) + 1e-12)
            by[l].append(float((e / (np.linalg.norm(e) + 1e-12)) @ dn))
    return {c: np.mean(v) for c, v in by.items()}

wr = within_class(rem_emb, [l for l, _ in removed])
wk = within_class(kept_emb, [l for l, _ in kept_list])
wn = within_class(naive_emb, [l for l, _ in naive_samp])

def paired(a, b, label):
    common = sorted(set(a) & set(b))
    da = np.array([a[c] for c in common]); db = np.array([b[c] for c in common])
    t, p = stats.ttest_rel(da, db)
    print("%-28s classes=%d delta=%+.4f t=%.2f p=%.4g (a_mean=%.4f b_mean=%.4f)" % (label, len(common), (da - db).mean(), t, p, da.mean(), db.mean()), flush=True)
    return {"label": label, "n_classes": len(common), "delta": float((da - db).mean()), "t": float(t), "p": float(p)}

res = {"overall": {"removed": float(np.mean(f_rem)), "kept": float(np.mean(f_kep)), "naive": float(np.mean(f_nai))},
       "paired": [paired(wr, wk, "removed vs kept"), paired(wr, wn, "removed vs naive"), paired(wk, wn, "kept vs naive")]}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("DONE", flush=True)

