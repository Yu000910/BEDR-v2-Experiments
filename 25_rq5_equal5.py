import numpy as np, json
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/out/round5/", exist_ok=True)

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
os.chdir(os.path.join(BASE, 'data'))
feasible = json.load(open('feasible_classes.json'))
if isinstance(feasible, dict):
    feasible = list(feasible.keys())
fs = sorted(set(str(c) for c in feasible))

def load(path):
    z = np.load(path, allow_pickle=True)
    X = z['vectors'].astype(np.float32)
    y = [str(t) for t in z['labels']]
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X/n, y

def matched30(X, y, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in fs:
        ci = [i for i, t in enumerate(y) if t == c]
        idx.append(rng.choice(ci, size=30, replace=False) if len(ci) >= 30 else ci)
    ii = np.concatenate(idx)
    return X[ii], np.array([y[i] for i in ii])

datasets = {}
datasets['BASE'] = load('D_base_feasible.npz')
datasets['EDA'] = load('D_eda.npz')
datasets['SMOTE'] = load('D_smote.npz')
datasets['IPF'] = load('D_smote_ipf.npz')
datasets['RKNN'] = load('D_smote_rknn.npz')
z = np.load(os.path.join(BASE, 'out/round5/D_bedrv2_HYBRID_top100.npz'), allow_pickle=True)
classes = json.load(open(os.path.join(BASE, 'data/classes_sorted.json')))
Xb = z['vectors'].astype(np.float32); yb = z['labels'].astype(np.int64)
names = np.array([classes[i] for i in yb])
m = np.array([n in fs for n in names])
n = np.linalg.norm(Xb[m], axis=1, keepdims=True); n[n==0]=1
datasets['BEDR'] = (Xb[m]/n, [str(t) for t in names[m]])
Xte, yte = load('D_test_feasible.npz')
print('test rows:', len(yte), flush=True)

out = {}
per_seed = {}
for name, (X, y) in datasets.items():
    accs, macs = [], []
    for seed in [42, 43, 44, 45, 46]:
        Xs, ys = matched30(X, y, seed)
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
        clf.fit(Xs, ys)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        macs.append(f1_score(yte, pred, average='macro'))
    out[name] = {'acc_mean': round(float(np.mean(accs)), 4), 'acc_std': round(float(np.std(accs)), 4), 'macro_f1_mean': round(float(np.mean(macs)), 4), 'macro_f1_std': round(float(np.std(macs)), 4)}
    per_seed[name] = [round(float(x), 4) for x in macs]
    print(name, json.dumps(out[name]), flush=True)
for other in ['EDA', 'SMOTE', 'IPF', 'RKNN']:
    t, p = stats.ttest_rel(np.array(per_seed['BEDR']), np.array(per_seed[other]))
    print('BEDR vs', other, 'macro-f1 paired t: t=%.2f p=%.4g' % (t, p), flush=True)
json.dump({'results': out, 'per_seed_macro': per_seed}, open(os.path.join(BASE, 'results/rq5_equalbudget5.json'), 'w'), indent=1)
print('DONE', flush=True)
