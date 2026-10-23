# -*- coding: utf-8 -*-
"""熵筛选消融的分层检验：熵选 vs 顺序选 vs 随机选，在 全部/零样本/长尾/头部 四个子集上比较。"""
import os, csv, json, time
import numpy as np, torch
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = BASE + "/out/embed_cache_securebert.npz"
tok = AutoTokenizer.from_pretrained(BASE + "/../hf_models/SecureBERT")
mdl = AutoModel.from_pretrained(BASE + "/../hf_models/SecureBERT").eval()

@torch.no_grad()
def _embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=160, return_tensors="pt")
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h*m).sum(1)/m.sum(1), dim=1)
        out.append(e.numpy().astype(np.float32))
    return np.vstack(out)

classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}
graph = json.load(open(BASE + "/data/intent_action_entity.json"))
zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
tB = [str(t) for t in zb["texts"]]; yB = np.array([n2i[str(x)] for x in zb["labels"]], dtype=np.int64)
cands = []
for f in ["round5","round6","round6b","round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if not os.path.exists(p): continue
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try: ms=float(r["maxsim"])
            except Exception: ms=0.0
            cands.append({"order":len(cands),"tech":r["tech_id"].strip(),"s":r["sentence"].strip(),"ms":ms})
tI = [str(graph.get(c,{}).get("intent","") or c) for c in classes]
zr = np.load(BASE + "/../exp/bedr2/out/round5/test_real_texts.npz", allow_pickle=True)
tR = [str(t) for t in zr["texts"]]
if os.path.exists(CACHE):
    z = np.load(CACHE); XB, VI, VC, XR = z["XB"], z["VI"], z["VC"], z["XR"]
    print("用缓存嵌入", flush=True)
else:
    t0=time.time(); XB=_embed(tB); VI=_embed(tI); VC=_embed([c["s"] for c in cands]); XR=_embed(tR)
    np.savez(CACHE, XB=XB, VI=VI, VC=VC, XR=XR); print("embed", round(time.time()-t0),"s", flush=True)
z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True)
m = np.asarray(z2["src"]) == "real"; yR = np.asarray(z2["labels"], dtype=np.int64)[m]
XBh = XB + VI[yB]; XBh /= np.linalg.norm(XBh, axis=1, keepdims=True)
yc = np.array([n2i[c["tech"]] if c["tech"] in n2i else 0 for c in cands])
VCh = VC + VI[yc]; VCh /= np.linalg.norm(VCh, axis=1, keepdims=True)
K=15; Sim=VCh@XBh.T; nb=np.argpartition(-Sim,K,axis=1)[:,:K]; labs=yB[nb]
h=np.zeros(len(cands),np.float32)
for j in range(len(labs)):
    v,c=np.unique(labs[j],return_counts=True); p=c/c.sum(); h[j]=-(p*np.log2(p)).sum()
by=defaultdict(list)
for j,c in enumerate(cands):
    if c["tech"] in n2i: by[c["tech"]].append(j)
cntB=Counter(yB.tolist()); TARGET=30
def select(mode, seed=42):
    rng=np.random.RandomState(seed); idx=[]
    for t,js in by.items():
        c=n2i[t]; need=max(0,TARGET-cntB.get(c,0))
        if need==0: continue
        if mode=="ent": order=sorted(js,key=lambda j:h[j])
        elif mode=="order": order=sorted(js,key=lambda j:cands[j]["order"])
        else: order=list(rng.permutation(js))
        idx.extend(order[:need])
    return np.array(sorted(set(idx)))
sel={m2:select(m2) for m2 in ["ent","order","rand"]}
print("新增: ent %d order %d rand %d | 熵选中与顺序选重合 %d 条（%.1f%%）"%(
 len(sel["ent"]),len(sel["order"]),len(sel["rand"]),
 len(np.intersect1d(sel["ent"],sel["order"])),100*len(np.intersect1d(sel["ent"],sel["order"]))/len(sel["ent"])),flush=True)
def build(ai,seed=42):
    X=np.vstack([XB,VC[ai]]); y=np.concatenate([yB,yc[ai]])
    rng=np.random.RandomState(seed); keep=[]
    for c in range(679):
        ci=np.where(y==c)[0]; keep.append(ci if len(ci)<=100 else rng.choice(ci,size=100,replace=False))
    k=np.sort(np.concatenate(keep)); return X[k],y[k]
zs=set(json.load(open(BASE+"/data/zero_shot_classes.json"))["zero_shot"]) if os.path.exists(BASE+"/data/zero_shot_classes.json") else set()
zs_idx=set(n2i[c] for c in zs if c in n2i)
cB=Counter(yB.tolist()); cT=Counter(yR.tolist())
tail=set(c for c in range(679) if cB.get(c,0)<30 and cT.get(c,0)>0)
masks={"全部(679类)":np.ones(len(yR),bool)}
if zs_idx and np.isin(yR,list(zs_idx)).sum()>0: masks["零样本(105类)"]=np.isin(yR,list(zs_idx))
masks["长尾(322类)"]=np.isin(yR,list(tail))
masks["头部(非长尾)"]=~masks["长尾(322类)"]
print("\n子集 | 样本数 | ent MLP / Ridge | order MLP / Ridge | rand MLP / Ridge")
Xr={}
for m2 in ["ent","order","rand"]: Xr[m2]=build(sel[m2])
for name,mk in masks.items():
    if mk.sum()==0: print(name,"：真实语句测试集中无该类测试句，跳过",flush=True); continue
    row=[]
    for m2 in ["ent","order","rand"]:
        X,y=Xr[m2]; a=[]
        for s in [42,43,44]:
            clf=MLPClassifier(hidden_layer_sizes=(256,),max_iter=20,early_stopping=True,n_iter_no_change=5,batch_size=256,learning_rate_init=1e-3,random_state=s).fit(X,y)
            a.append(accuracy_score(yR[mk],clf.predict(XR[mk])))
        rg=RidgeClassifier(alpha=1.0).fit(X,y); row.append((np.mean(a),accuracy_score(yR[mk],rg.predict(XR[mk]))))
    print(f"{name:12s} {int(mk.sum()):5d} | {row[0][0]:.4f} / {row[0][1]:.4f} | {row[1][0]:.4f} / {row[1][1]:.4f} | {row[2][0]:.4f} / {row[2][1]:.4f}",flush=True)
