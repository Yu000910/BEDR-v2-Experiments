import numpy as np, json
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}

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

XB, yB = load(BASE + '/data/D_base_train_new.npz', 'name')
XN, yN = load(BASE + '/data/D_naive_full.npz', 'name')
XE, yE = load(BASE + '/out/round5/D_bedr_HYBRID_top100.npz', 'int')
z2 = np.load(BASE + '/data/test2_full_plain.npz', allow_pickle=True)
m = np.asarray(z2['src']) == 'real'
XT = norm(z2['vectors'][m].astype(np.float32)); yT = np.asarray(z2['labels'], dtype=np.int64)[m]

cB, cN, cE = Counter(yB.tolist()), Counter(yN.tolist()), Counter(yE.tolist())
out = {}
for k in [5, 10, 30]:
    elig = sorted([c for c in set(cB) & set(cN) & set(cE) if cB[c] >= k and cN[c] >= k and cE[c] >= k and c in set(yT.tolist())])
    mask = np.isin(yT, elig)
    Xte, yte = XT[mask], yT[mask]
    for jname in ['MLP', 'RIDGE']:
        row = {}
        for name, (X, y) in [('BASE', (XB, yB)), ('NAIVE', (XN, yN)), ('BEDR', (XE, yE))]:
            accs = []
            for seed in [42, 43, 44, 45, 46]:
                rng = np.random.RandomState(seed)
                parts = []
                for c in elig:
                    ci = np.where(y == c)[0]
                    parts.append(rng.choice(ci, size=k, replace=False))
                idx = np.concatenate(parts)
                if jname == 'MLP':
                    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
                else:
                    clf = RidgeClassifier(alpha=1.0, random_state=seed)
                clf.fit(X[idx], y[idx])
                accs.append(accuracy_score(yte, clf.predict(Xte)))
            row[name] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
        out['k%d_%s' % (k, jname)] = row
        print('k=%d %s elig=%d' % (k, jname, len(elig)), json.dumps(row), flush=True)
json.dump(out, open(os.path.join(BASE, 'results/common_matched.json'), 'w'), indent=1)
print('DONE', flush=True)
