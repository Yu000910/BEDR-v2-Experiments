import numpy as np, json, csv, os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE + '/out/round5'
os.makedirs(BASE + '/out/round5', exist_ok=True)
classes_sorted = json.load(open(BASE + '/data/classes_sorted.json'))
name2idx = {c: i for i, c in enumerate(classes_sorted)}

# ---- pool0 with texts ----
zb = np.load(BASE + '/data/D_base_train.npz', allow_pickle=True)
XB = zb['vectors'].astype(np.float32); yB = np.array([name2idx[str(t)] for t in zb['labels']], dtype=np.int64); tB = np.array([str(t) for t in zb['texts']])
emb = np.load(BASE + '/out/fidelity_audit/embeds.npz')
txts = json.load(open(BASE + '/out/fidelity_audit/texts.json'))
rem_sents = [t for _, t in txts['removed']]; kep_sents = [t for _, t in txts['kept']]
yC = np.array([name2idx[t] for t, _ in txts['removed']] + [name2idx[t] for t, _ in txts['kept']], dtype=np.int64)
XC = np.vstack([emb['rem_emb'], emb['kept_emb']]).astype(np.float32)
tC = np.array(rem_sents + kep_sents)
fd = np.load(BASE + '/out/filter_corrected/filter_decision.npz')
acc10 = fd['accept_local10']
pool_X = np.vstack([XB, XC[acc10]]); pool_y = np.concatenate([yB, yC[acc10]]); pool_t = np.concatenate([tB, tC[acc10]])
cnt0 = Counter(pool_y.tolist())
print('pool0:', pool_X.shape, flush=True)

def read_cands(fn):
    rows = []
    with open(OUT + '/' + fn, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append({'tech': r['tech_id'], 'sentence': r['sentence'], 'hybrid_ent': float(r['hybrid_ent']), 'maxsim': float(r['maxsim'])})
    return rows

# ---- R5 HYBRID selection (mirror build_round5_arms select) ----
cands5 = read_cands('round5_candidates.csv')
by5 = {}
for r in cands5:
    by5.setdefault(r['tech'], []).append(r)
sel5 = []  # (sentence) in append order
for t, rs in by5.items():
    c = name2idx[t]
    need = max(0, 30 - cnt0.get(c, 0))
    taken = 0
    for r in sorted(rs, key=lambda r: r['hybrid_ent']):
        if taken >= need: break
        if r['maxsim'] >= 0.9: continue
        if r['hybrid_ent'] < 0 or r['hybrid_ent'] > 2.3: continue
        sel5.append(r['sentence']); taken += 1
print('R5 accepted:', len(sel5), flush=True)

# ---- R6 + fill (mirror build_round6_arms + fill_underfilled) ----
cands6 = read_cands('round6_candidates.csv') + read_cands('round6b_candidates.csv')
by6 = {}
for r in cands6:
    by6.setdefault(r['tech'], []).append(r)
sel6 = []
for t, rs in by6.items():
    c = name2idx[t]
    need = max(0, 30 - cnt0.get(c, 0))
    taken = 0
    for r in sorted(rs, key=lambda r: r['hybrid_ent']):
        if taken >= need: break
        if r['maxsim'] >= 0.9: continue
        if r['hybrid_ent'] < 0 or r['hybrid_ent'] > 2.3: continue
        sel6.append(r['sentence']); taken += 1
print('R6 accepted:', len(sel6), flush=True)

cnt_mid = Counter(pool_y.tolist())
cnt_mid.update([name2idx[t] for t, _ in [(s.split(chr(0))[0], '') for s in []]])  # noop keep structure
sel_mid = sel5 + sel6
cnt_after = Counter(cnt0)
for s in sel_mid:
    pass
cnt_after = Counter(pool_y.tolist())
for r in sel5:
    pass
# recompute counts after R5+R6 by class from sel lists directly
cnt_mid = Counter(pool_y.tolist())
for r in cands5:
    pass
# build class counts of sel5/sel6
def count_sel(sel, cands_by):
    pass
cnt_after56 = Counter(pool_y.tolist())
for s in sel5:
    c = name2idx[[r['tech'] for r in cands5 if r['sentence'] == s][0]]
    cnt_after56[c] += 1
for s in sel6:
    c = name2idx[[r['tech'] for r in cands6 if r['sentence'] == s][0]]
    cnt_after56[c] += 1
# fill: classes under 30
fill_add = []
for c in range(679):
    if cnt_after56.get(c, 0) >= 30: continue
    t = classes_sorted[c]
    rs = sorted(by6.get(t, []), key=lambda r: r['hybrid_ent'])
    n = cnt_after56.get(c, 0)
    for r in rs:
        if n >= 30: break
        if r['maxsim'] >= 0.9: continue
        fill_add.append(r['sentence']); n += 1
print('fill added:', len(fill_add), flush=True)

# ---- R7 (mirror gen_round7 fill part) ----
cands7 = read_cands('round7_candidates.csv')
by7 = {}
for r in cands7:
    by7.setdefault(r['tech'], []).append(r)
cnt_after_fill = Counter(cnt_after56)
for s in fill_add:
    c = name2idx[[r['tech'] for r in cands6 if r['sentence'] == s][0]]
    cnt_after_fill[c] += 1
under = sorted([c for c in range(679) if cnt_after_fill.get(c, 0) < 30])
r7_add = []
for c in under:
    t = classes_sorted[c]
    n = cnt_after_fill.get(c, 0)
    for r in sorted(by7.get(t, []), key=lambda r: r['hybrid_ent']):
        if n >= 30: break
        if r['maxsim'] >= 0.92: continue
        r7_add.append(r['sentence']); n += 1
print('R7 added:', len(r7_add), flush=True)

# ---- assemble pre-TOP100: pool0 + sel5 + sel6 + fill_add + r7_add, paired with D_bedrv2_HYBRID.npz vectors ----
z5 = np.load(OUT + '/D_round5_HYBRID.npz', allow_pickle=True)
zfull = np.load(OUT + '/D_bedrv2_HYBRID.npz', allow_pickle=True)
print('D_round5_HYBRID:', z5['vectors'].shape, 'D_bedrv2_HYBRID(pre-top100):', zfull['vectors'].shape, flush=True)
V5 = z5['vectors'].astype(np.float32)
Vfull = zfull['vectors'].astype(np.float32)
assert len(pool_y) + len(sel5) == V5.shape[0], 'R5 size mismatch'
tail_texts = sel6 + fill_add + r7_add
assert len(tail_texts) == Vfull.shape[0] - V5.shape[0], 'tail size mismatch %d vs %d' % (len(tail_texts), Vfull.shape[0] - V5.shape[0])
preX = np.vstack([V5, Vfull[V5.shape[0]:]])
preY = np.concatenate([z5['labels'].astype(np.int64), zfull['labels'].astype(np.int64)[V5.shape[0]:]])
preT = np.concatenate([pool_t, np.array(sel5 + sel6 + fill_add + r7_add)])
print('pre-TOP100 assembled:', preX.shape, len(preT), flush=True)

# ---- TOP100 matching ----
zt = np.load(OUT + '/D_bedrv2_HYBRID_top100.npz', allow_pickle=True)
VT = zt['vectors'].astype(np.float32); yT = zt['labels'].astype(np.int64)
S = VT @ preX.T
idx = S.argmax(axis=1)
exact = (S.max(axis=1) > 0.999999).mean()
print('TOP100 exact match rate:', round(float(exact), 5), flush=True)
texts_T = preT[idx]
np.savez(OUT + '/D_bedrv2_HYBRID_top100_texts.npz', vectors=VT, labels=yT, texts=texts_T)
print('saved BEDR final with texts:', VT.shape, flush=True)

# ---- test texts ----
t5422 = np.load(BASE + '/data/orig5422_plain.npz')['vectors'].astype(np.float32)
tn = np.linalg.norm(t5422, axis=1, keepdims=True); tn[tn==0]=1
t5422n = t5422 / tn
r = np.load(BASE + '/out/test1_real_plain.npz', allow_pickle=True)
V = r['vectors'].astype(np.float32)
vn = np.linalg.norm(V, axis=1, keepdims=True); vn[vn==0]=1
S2 = (V/vn) @ t5422n.T
idx2 = S2.argmax(axis=1)
import pandas as pd
df = pd.read_csv(os.path.join(BASE, 'data/final_test_set.csv'))
col = 'sentence' if 'sentence' in df.columns else df.columns[-1]
test_texts = np.array([str(t) for t in df[col].values])[idx2]
np.savez(OUT + '/test_real_texts.npz', vectors=V, labels=r['labels'].astype(np.int64), texts=test_texts)
print('saved test texts:', V.shape, flush=True)
print('sample test text:', test_texts[0][:100], flush=True)
print('sample BEDR text:', texts_T[0][:100], flush=True)
print('DONE', flush=True)
