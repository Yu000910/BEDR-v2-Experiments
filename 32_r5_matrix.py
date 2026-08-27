import numpy as np, json
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
cohort = json.load(open(BASE + '/out/fix_pilot/tail_cohort.json'))['tail']
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n
z = np.load(BASE + '/out/round5/D_round5_HYBRID.npz', allow_pickle=True)
X = norm(z['vectors'].astype(np.float32)); y = z['labels'].astype(np.int64)
z2 = np.load(BASE + '/data/test2_full_plain.npz', allow_pickle=True)
m = np.asarray(z2['src']) == 'real'
XT = norm(z2['vectors'][m].astype(np.float32)); yT = np.asarray(z2['labels'], dtype=np.int64)[m]
out = {}
for jname in ['MLP', 'RIDGE']:
    accs = []
    for seed in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed) if jname == 'MLP' else RidgeClassifier(alpha=1.0, random_state=seed)
        clf.fit(X, y)
        accs.append(accuracy_score(yT, clf.predict(XT)))
    out['full_' + jname] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
tmask = np.isin(yT, cohort)
Xtt, ytt = XT[tmask], yT[tmask]
class CosNC:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        C = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.W_ = C.astype(np.float32)
        return self
    def predict(self, X):
        idx = (X.astype(np.float32) @ self.W_.T).argmax(axis=1)
        return self.classes_[idx]
def matched(X, y, k, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for c in cohort:
        ci = np.where(y == c)[0]
        parts.append(rng.choice(ci, size=k, replace=False) if len(ci) >= k else ci)
    return X[np.concatenate(parts)], y[np.concatenate(parts)]
for k in [10, 20, 30]:
    for jname in ['NC', 'RIDGE']:
        accs = []
        for seed in [42, 43, 44]:
            Xs, ys = matched(X, y, k, seed)
            clf = CosNC().fit(Xs, ys) if jname == 'NC' else RidgeClassifier(alpha=1.0, random_state=seed).fit(Xs, ys)
            accs.append(accuracy_score(ytt, clf.predict(Xtt)))
        out['tail_k%d_%s' % (k, jname)] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
print(json.dumps(out))
json.dump(out, open(BASE + '/results/r5_matrix.json', 'w'), indent=1)
print('DONE')
