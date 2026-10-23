import numpy as np, json
from scipy import stats
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
zs = set(json.load(open(BASE + '/data/zero_shot_classes.json'))['zero_shot'])
zs_idx = [name2idx[c] for c in zs]
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
XE, yE = loady(BASE + '/out/round5/D_bedr_HYBRID_top100.npz', 'int')
XN, yN = loady(BASE + '/data/D_naive_full.npz', 'name')
emb = np.load(BASE + '/out/fidelity_audit/embeds.npz')
defn = {}
for l, v in zip(emb['def_labels'], emb['def_emb']):
    defn[name2idx[str(l)]] = v
def align(X, y):
    out = []
    for c in zs_idx:
        mm = y == c
        if mm.sum() == 0:
            continue
        cent = X[mm].mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)
        dn = defn[c] / (np.linalg.norm(defn[c]) + 1e-12)
        out.append(float(cent @ dn))
    return np.array(out)
aE = align(XE, yE)
aN = align(XN, yN)
t, p = stats.ttest_rel(aE, aN)
out = {'bedr_mean': round(float(aE.mean()), 4), 'naive_mean': round(float(aN.mean()), 4), 't': round(float(t), 2), 'p': float(p), 'n': int(len(aE))}
print(json.dumps(out))
json.dump(out, open(BASE + '/results/zero_align.json', 'w'), indent=1)
print('DONE')
