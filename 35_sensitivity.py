import numpy as np, json
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n
zb = np.load(BASE + '/data/D_base_train.npz', allow_pickle=True)
XB = norm(zb['vectors']); yB = np.array([name2idx[str(t)] for t in zb['labels']], dtype=np.int64)
emb = np.load(BASE + '/out/fidelity_audit/embeds.npz')
txts = json.load(open(BASE + '/out/fidelity_audit/texts.json'))
yC = np.array([name2idx[t] for t, _ in txts['removed']] + [name2idx[t] for t, _ in txts['kept']], dtype=np.int64)
XC = np.vstack([norm(emb['rem_emb']), norm(emb['kept_emb'])])
XP = np.vstack([XB, XC]); yP = np.concatenate([yB, yC])
nB = len(yB); nC = len(yC); nP = len(yP)
z2 = np.load(BASE + '/data/test2_full_plain.npz', allow_pickle=True)
m = np.asarray(z2['src']) == 'real'
XT = norm(z2['vectors'][m].astype(np.float32)); yT = np.asarray(z2['labels'], dtype=np.int64)[m]
print('pool', nP, flush=True)

def local_entropy_all(K):
    h = np.zeros(nP, dtype=np.float32)
    for i in range(0, nP, 2048):
        D = 1.0 - XP[i:i+2048] @ XP.T
        r0 = np.arange(i, min(i+2048, nP))
        D[np.arange(len(r0)), r0] = 1.0
        part = np.argpartition(D, K-1, axis=1)[:, :K]
        labs = yP[part]
        for r in range(len(part)):
            vals, counts = np.unique(labs[r], return_counts=True)
            p = counts / counts.sum()
            h[i+r] = -(p * np.log2(p)).sum()
    return h

out = {}
for K in [5, 15, 30]:
    h = local_entropy_all(K)
    variants = {'no_filter': np.ones(nC, dtype=bool)}
    for th in [1.0, 1.5, 2.3]:
        variants['theta_%.1f' % th] = h[nB:] < th
    for vname, mask in variants.items():
        Xd = np.vstack([XB, XC[mask]]); yd = np.concatenate([yB, yC[mask]])
        accs = []
        for seed in [42, 43, 44]:
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
            clf.fit(Xd, yd)
            accs.append(accuracy_score(yT, clf.predict(XT)))
        key = 'k%d_%s' % (K, vname)
        out[key] = {'n_train': int(len(yd)), 'acc_mean': round(float(np.mean(accs)), 4), 'acc_std': round(float(np.std(accs)), 4)}
        print(key, json.dumps(out[key]), flush=True)
json.dump(out, open(BASE + '/results/sensitivity.json', 'w'), indent=1)
print('DONE', flush=True)
