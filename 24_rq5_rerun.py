import numpy as np, json, time
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
feasible_s = set(str(c) for c in feasible)
print('feasible classes:', len(feasible_s), flush=True)

def load(path):
    z = np.load(path, allow_pickle=True)
    X = z['vectors'].astype(np.float32)
    y = [str(t) for t in z['labels']]
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X/n, y

datasets = {}
datasets['BASE'] = load('D_base_feasible.npz')
datasets['EDA'] = load('D_eda.npz')
datasets['SMOTE'] = load('D_smote.npz')
datasets['IPF'] = load('D_smote_ipf.npz')
datasets['RKNN'] = load('D_smote_rknn.npz')
# BEDR-v2 feasible
z = np.load(os.path.join(BASE, 'out/round5/D_bedrv2_HYBRID_top100.npz'), allow_pickle=True)
classes = json.load(open(os.path.join(BASE, 'data/classes_sorted.json')))
Xb = z['vectors'].astype(np.float32); yb = z['labels'].astype(np.int64)
names = np.array([classes[i] for i in yb])
m = np.array([n in feasible_s for n in names])
Xb2 = Xb[m]; yb2 = [str(t) for t in names[m]]
n = np.linalg.norm(Xb2, axis=1, keepdims=True); n[n==0]=1
datasets['BEDRv2'] = (Xb2/n, yb2)
print('BEDR-v2 feasible rows:', len(yb2), flush=True)

Xte, yte = load('D_test_feasible.npz')
print('test rows:', len(yte), flush=True)

results = {}
for name, (X, y) in datasets.items():
    accs, macs = [], []
    for seed in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5, batch_size=256, learning_rate_init=1e-3, random_state=seed)
        clf.fit(X, y)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        macs.append(f1_score(yte, pred, average='macro'))
    results[name] = {'acc_mean': round(float(np.mean(accs)), 4), 'acc_std': round(float(np.std(accs)), 4), 'macro_f1_mean': round(float(np.mean(macs)), 4), 'macro_f1_std': round(float(np.std(macs)), 4)}
    print(name, json.dumps(results[name]), flush=True)
json.dump(results, open(os.path.join(BASE, 'results/rq5_rerun.json'), 'w'), indent=1)
print('DONE', flush=True)
