import numpy as np, json
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
fam_names = [c for c in classes if c == 'T1546' or c.startswith('T1546.')]
fam_idx = [name2idx[c] for c in fam_names]
print('T1546 family:', fam_names, len(fam_names), flush=True)

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n

def load(path, kind):
    z = np.load(path, allow_pickle=True)
    X = norm(z['vectors'].astype(np.float32))
    y_raw = z['labels']
    if y_raw.dtype.kind in 'UO':
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y

datasets = {
    'BASE': load(BASE + '/data/D_base_train.npz', 'name'),
    'NAIVE': load(BASE + '/data/D_naive_full.npz', 'name'),
    'BEDR': load(BASE + '/out/round5/D_bedrv2_HYBRID_top100.npz', 'int'),
}

def metrics(X, y):
    m = np.isin(y, fam_idx)
    Xf, yf = X[m], y[m]
    # boundary entropy (k=15, plain, self-excluded)
    K = 15
    h = np.zeros(len(Xf), dtype=np.float32)
    TOP = 16
    for i in range(0, len(Xf), 2048):
        D = 1.0 - Xf[i:i+2048] @ Xf.T
        D[i:i+2048, i:i+2048] = 1.0
        part = np.argpartition(D, TOP-1, axis=1)[:, :TOP]
        idx = part[:, :K]
        labs = yf[idx]
        for r in range(len(idx)):
            vals, counts = np.unique(labs[r], return_counts=True)
            p = counts / counts.sum()
            h[i+r] = -(p * np.log2(p)).sum()
    ent = float(h.mean())
    sil = float(silhouette_score(Xf, yf, metric='cosine'))
    db = float(davies_bouldin_score(Xf, yf))
    # intra: mean pairwise (1-cos) per class, averaged
    intra = []
    cents = {}
    for c in fam_idx:
        Xc = Xf[yf == c]
        if len(Xc) < 2:
            continue
        S = Xc @ Xc.T
        ii, jj = np.triu_indices(len(Xc), k=1)
        intra.append(float((1.0 - S[ii, jj]).mean()))
        cents[c] = Xc.mean(axis=0)
    intra_m = float(np.mean(intra))
    # inter: mean (1-cos) between centroids of distinct classes
    keys = sorted(cents.keys())
    inter = []
    for a in range(len(keys)):
        for b in range(a+1, len(keys)):
            ca = cents[keys[a]] / (np.linalg.norm(cents[keys[a]])+1e-12)
            cb = cents[keys[b]] / (np.linalg.norm(cents[keys[b]])+1e-12)
            inter.append(float(1.0 - ca @ cb))
    inter_m = float(np.mean(inter))
    sep = inter_m / (intra_m + 1e-12)
    return {'n': int(len(Xf)), 'boundary_entropy': round(ent, 4), 'silhouette': round(sil, 4), 'davies_bouldin': round(db, 4), 'intra_cos_dist': round(intra_m, 4), 'inter_cos_dist': round(inter_m, 4), 'separation_ratio': round(sep, 4)}

out = {}
for name, (X, y) in datasets.items():
    r = metrics(X, y)
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open(os.path.join(BASE, 'results/six_metrics.json'), 'w'), indent=1)
print('DONE', flush=True)
