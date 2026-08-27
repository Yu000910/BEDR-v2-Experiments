import numpy as np, json
from collections import Counter
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
def loady(path, kind):
    z = np.load(path, allow_pickle=True)
    y_raw = z['labels']
    if y_raw.dtype.kind in 'UO':
        y = np.array([name2idx[str(t)] for t in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return y
yB = loady(BASE + '/data/D_base_train_new.npz', 'name')
yN = loady(BASE + '/data/D_naive_full.npz', 'name')
yE = loady(BASE + '/out/round5/D_bedrv2_HYBRID_top100.npz', 'int')
z2 = np.load(BASE + '/data/test2_full_plain.npz', allow_pickle=True)
m = np.asarray(z2['src']) == 'real'
yT = np.asarray(z2['labels'], dtype=np.int64)[m]
def dist(y):
    c = Counter(y.tolist())
    p = np.zeros(679)
    for k, v in c.items():
        p[k] = v
    return p / p.sum()
pB, pN, pE, pT = dist(yB), dist(yN), dist(yE), dist(yT)
def js(p, q):
    M = (p + q) / 2
    def kl(a, b):
        mm = (a > 0) & (b > 0)
        return float((a[mm] * np.log(a[mm] / b[mm])).sum())
    return 0.5 * kl(p, M) + 0.5 * kl(q, M)
out = {'baseline': round(js(pB, pT), 4), 'naive': round(js(pN, pT), 4), 'bedr': round(js(pE, pT), 4)}
print(json.dumps(out))
json.dump(out, open(BASE + '/results/js_div.json', 'w'), indent=1)
print('DONE')
