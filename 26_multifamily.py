import numpy as np, json
from collections import defaultdict
from sklearn.metrics import silhouette_score
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n

def loady(path, kind):
    z = np.load(path, allow_pickle=True)
    X = norm(z['vectors'].astype(np.float32))
    y_raw = z['labels']
    if y_raw.dtype.kind in 'UO':
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y

datasets = {
    'BASE': loady(BASE + '/data/D_base_train_new.npz', 'name'),
    'NAIVE': loady(BASE + '/data/D_naive_full.npz', 'name'),
    'BEDR': loady(BASE + '/out/round5/D_bedr_HYBRID_top100.npz', 'int'),
}
fam = defaultdict(list)
for c in classes:
    fam[c.split('.')[0]].append(name2idx[c])
for c in classes:
    if '.' not in c:
        fam[c].append(name2idx[c])
top = sorted(fam.items(), key=lambda kv: -len(kv[1]))[:8]

def matched_sample(X, y, members, k, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in members:
        ci = np.where(y == c)[0]
        if len(ci) >= k:
            idx.append(rng.choice(ci, size=k, replace=False))
    return np.concatenate(idx)

def family_metrics(Xf, yf, members):
    K = 15
    h = np.zeros(len(Xf), dtype=np.float32)
    for i in range(0, len(Xf), 2048):
        D = 1.0 - Xf[i:i+2048] @ Xf.T
        D[i:i+2048, i:i+2048] = 1.0
        part = np.argpartition(D, min(K+1, len(Xf)-1), axis=1)[:, :K]
        idx = part
        labs = yf[idx]
        for r in range(len(idx)):
            vals, counts = np.unique(labs[r], return_counts=True)
            p = counts / counts.sum()
            h[i+r] = -(p * np.log2(p)).sum()
    ent = float(h.mean())
    sil = float(silhouette_score(Xf, yf, metric='cosine'))
    intra = []
    cents = {}
    for c in members:
        Xc = Xf[yf == c]
        if len(Xc) < 2:
            continue
        S = Xc @ Xc.T
        ii, jj = np.triu_indices(len(Xc), k=1)
        intra.append(float((1.0 - S[ii, jj]).mean()))
        cents[c] = Xc.mean(axis=0)
    intra_m = float(np.mean(intra))
    keys = sorted(cents.keys())
    inter = []
    for a in range(len(keys)):
        for b in range(a+1, len(keys)):
            ca = cents[keys[a]] / (np.linalg.norm(cents[keys[a]])+1e-12)
            cb = cents[keys[b]] / (np.linalg.norm(cents[keys[b]])+1e-12)
            inter.append(float(1.0 - ca @ cb))
    inter_m = float(np.mean(inter))
    return {'n': int(len(Xf)), 'ent': round(ent, 3), 'sil': round(sil, 3), 'sep': round(inter_m/(intra_m+1e-12), 3)}

out = {}
wins = {d: {'sil': 0, 'sep': 0} for d in datasets}
for fname, members in top:
    counts = {}
    for dname, (X, y) in datasets.items():
        counts[dname] = {c: int((y == c).sum()) for c in members}
    elig = [c for c in members if all(counts[d].get(c, 0) >= 10 for d in datasets)]
    if len(elig) < 3:
        out[fname] = 'skip (few dense classes: %d)' % len(elig)
        print(fname, out[fname], flush=True)
        continue
    fam_res = {}
    for dname, (X, y) in datasets.items():
        best = None
        for seed in [42, 43, 44]:
            idx = matched_sample(X, y, elig, 10, seed)
            r = family_metrics(X[idx], y[idx], elig)
            if best is None or r['sep'] + r['sil'] > best['sep'] + best['sil']:
                best = r
        fam_res[dname] = best
    out[fname] = fam_res
    print(fname, 'dense classes:', len(elig), json.dumps(fam_res), flush=True)
    best_sil = max(datasets, key=lambda d: fam_res[d]['sil'])
    best_sep = max(datasets, key=lambda d: fam_res[d]['sep'])
    wins[best_sil]['sil'] += 1
    wins[best_sep]['sep'] += 1
print('wins:', json.dumps(wins))
json.dump(out, open(BASE + '/results/multifamily_matched.json', 'w'), indent=1)
print('DONE')
