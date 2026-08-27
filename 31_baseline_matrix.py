import numpy as np, json
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n
zb = np.load(BASE + '/data/D_base_train_new.npz', allow_pickle=True)
XB = norm(zb['vectors'].astype(np.float32)); yB = np.array([name2idx[str(t)] for t in zb['labels']], dtype=np.int64)
z2 = np.load(BASE + '/data/test2_full_plain.npz', allow_pickle=True)
m = np.asarray(z2['src']) == 'real'
XT = norm(z2['vectors'][m].astype(np.float32)); yT = np.asarray(z2['labels'], dtype=np.int64)[m]
out = {}
for jname in ['MLP', 'RIDGE']:
    accs = []
    for seed in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed) if jname == 'MLP' else RidgeClassifier(alpha=1.0, random_state=seed)
        clf.fit(XB, yB)
        accs.append(accuracy_score(yT, clf.predict(XT)))
    out['full_' + jname] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
cnt = Counter(yB.tolist())
for k in [5, 10, 30]:
    elig = sorted([c for c in cnt if cnt[c] >= k and c in set(yT.tolist())])
    mask = np.isin(yT, elig)
    Xte, yte = XT[mask], yT[mask]
    for jname in ['MLP', 'RIDGE']:
        accs = []
        for seed in [42, 43, 44]:
            rng = np.random.RandomState(seed)
            parts = []
            for c in elig:
                ci = np.where(yB == c)[0]
                parts.append(rng.choice(ci, size=k, replace=False))
            idx = np.concatenate(parts)
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed) if jname == 'MLP' else RidgeClassifier(alpha=1.0, random_state=seed)
            clf.fit(XB[idx], yB[idx])
            accs.append(accuracy_score(yte, clf.predict(Xte)))
        out['matched_k%d_%s' % (k, jname)] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4), len(elig)]
print(json.dumps(out))
json.dump(out, open(BASE + '/results/baseline_matrix.json', 'w'), indent=1)
print('DONE')
