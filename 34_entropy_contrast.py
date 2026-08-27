import numpy as np, json
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE + "/results", exist_ok=True)
PROJ = os.path.join(BASE, "data")
classes = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X.astype(np.float32)/n
def mean_entropy(V, y, k=15):
    h = np.zeros(len(V), dtype=np.float32)
    TOP = 16
    for i in range(0, len(V), 2048):
        D = 1.0 - V[i:i+2048] @ V.T
        D[i:i+2048, i:i+2048] = 1.0
        part = np.argpartition(D, TOP-1, axis=1)[:, :TOP]
        idx = part[:, :k]
        labs = y[idx]
        for r in range(len(idx)):
            vals, counts = np.unique(labs[r], return_counts=True)
            p = counts / counts.sum()
            h[i+r] = -(p * np.log2(p)).sum()
    return float(h.mean())
out = {}
zb = np.load(BASE + '/data/D_base_train_new.npz', allow_pickle=True)
XB = norm(zb['vectors']); yB = np.array([name2idx[str(t)] for t in zb['labels']], dtype=np.int64)
emb = np.load(BASE + '/out/fidelity_audit/embeds.npz')
txts = json.load(open(BASE + '/out/fidelity_audit/texts.json'))
yC = np.array([name2idx[t] for t, _ in txts['removed']] + [name2idx[t] for t, _ in txts['kept']], dtype=np.int64)
XC = np.vstack([norm(emb['rem_emb']), norm(emb['kept_emb'])])
XP = np.vstack([XB, XC]); yP = np.concatenate([yB, yC])
out['plain_pool_mean_entropy'] = round(mean_entropy(XP, yP), 4)
zh = np.load(PROJ + 'all_samples_intent_sentence_embed.npz', allow_pickle=True)
Xh = norm(zh['vectors']); yh = np.array([name2idx[str(t)] for t in zh['labels']], dtype=np.int64)
out['hybrid_pool_mean_entropy'] = round(mean_entropy(Xh, yh), 4)
fd = np.load(BASE + '/out/filter_corrected/filter_decision.npz')
Xf = np.vstack([XB, XC[fd['accept_local10']]]); yf = np.concatenate([yB, yC[fd['accept_local10']]])
zint = np.load(BASE + '/data/intent_embeds.npz', allow_pickle=True)
intent_by_name = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zint['labels'], zint['vectors'])}
I = np.stack([intent_by_name[classes[c]] for c in yf])
sweep = {}
for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    V = norm((1.0 - alpha) * Xf + alpha * I)
    sweep[str(alpha)] = round(mean_entropy(V, yf), 4)
out['anchor_sweep'] = sweep
print(json.dumps(out))
json.dump(out, open(BASE + '/results/entropy_contrast.json', 'w'), indent=1)
print('DONE')
