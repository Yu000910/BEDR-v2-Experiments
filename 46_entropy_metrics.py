# -*- coding: utf-8 -*-
"""用更敏感的指标检验熵筛选：难样本子集、父子混淆子集、Macro-F1、Top-3、NLL、Brier"""
import os, csv, json
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
BASE = os.path.dirname(os.path.abspath(__file__))
z = np.load(BASE + "/out/embed_cache_securebert.npz")
XB, VI, VC, XR = z["XB"], z["VI"], z["VC"], z["XR"]
classes = json.load(open(BASE + "/data/classes_sorted.json")); n2i = {c: i for i, c in enumerate(classes)}
zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
yB = np.array([n2i[str(x)] for x in zb["labels"]], dtype=np.int64)
cands = []
for f in ["round5","round6","round6b","round7"]:
    p = BASE + f"/data/candidates/{f}_candidates.csv"
    if not os.path.exists(p): continue
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cands.append({"order":len(cands),"tech":r["tech_id"].strip(),"s":r["sentence"].strip()})
yc = np.array([n2i[c["tech"]] if c["tech"] in n2i else 0 for c in cands])
XBh = XB + VI[yB]; XBh /= np.linalg.norm(XBh,axis=1,keepdims=True)
VCh = VC + VI[yc]; VCh /= np.linalg.norm(VCh,axis=1,keepdims=True)
K=15; Sim=VCh@XBh.T; nb=np.argpartition(-Sim,K,axis=1)[:,:K]; labs=yB[nb]
h=np.zeros(len(cands),np.float32)
for j in range(len(labs)):
    v,c=np.unique(labs[j],return_counts=True); p=c/c.sum(); h[j]=-(p*np.log2(p)).sum()
by=defaultdict(list)
for j,c in enumerate(cands):
    if c["tech"] in n2i: by[c["tech"]].append(j)
cntB=Counter(yB.tolist()); TARGET=30
def select(mode,seed=42):
    rng=np.random.RandomState(seed); idx=[]
    for t,js in by.items():
        c=n2i[t]; need=max(0,TARGET-cntB.get(c,0))
        if need==0: continue
        order=sorted(js,key=lambda j:h[j]) if mode=="ent" else (sorted(js,key=lambda j:cands[j]["order"]) if mode=="order" else list(rng.permutation(js)))
        idx.extend(order[:need])
    return np.array(sorted(set(idx)))
sel={m:select(m) for m in ["ent","order","rand"]}
z2=np.load(BASE+"/data/test2_full_plain.npz",allow_pickle=True); msk=np.asarray(z2["src"])=="real"
yR=np.asarray(z2["labels"],dtype=np.int64)[msk]
# 测试句在 SecureBERT 空间的 LBE（参考=基础集）
XRl = XR + VI[yR]; XRl /= np.linalg.norm(XRl,axis=1,keepdims=True)
S=XRl@XBh.T; nbr=np.argpartition(-S,K,axis=1)[:,:K]; lb=yB[nbr]
ht=np.zeros(len(yR),np.float32)
for j in range(len(lb)):
    v,c=np.unique(lb[j],return_counts=True); p=c/c.sum(); ht[j]=-(p*np.log2(p)).sum()
q75=np.percentile(ht,75)
masks={"全部":np.ones(len(yR),bool),"最难25%（LBE最高）":ht>=q75}
# 父子混淆：测试句的父技术与其自身不同类（同一父下有多个子技术）
parent=defaultdict(set)
for c in classes:
    if "." in c: parent[c.split(".")[0]].add(n2i[c])
sib_fam=set(c for c in range(679) if len(parent.get(classes[c].split(".")[0],[]))>=2)
masks["父子同族(有兄弟类)"]=np.isin(yR,list(sib_fam))
def build(ai,seed=42):
    X=np.vstack([XB,VC[ai]]); y=np.concatenate([yB,yc[ai]])
    rng=np.random.RandomState(seed); keep=[]
    for c in range(679):
        ci=np.where(y==c)[0]; keep.append(ci if len(ci)<=100 else rng.choice(ci,size=100,replace=False))
    k=np.sort(np.concatenate(keep)); return X[k],y[k]
print(f"{'指标':26s}{'熵选':>12s}{'顺序选':>12s}{'随机选':>12s}")
res={}
for m in ["ent","order","rand"]:
    X,y=build(sel[m]); clf=MLPClassifier(hidden_layer_sizes=(256,),max_iter=20,early_stopping=True,n_iter_no_change=5,batch_size=256,learning_rate_init=1e-3,random_state=42).fit(X,y)
    prob=clf.predict_proba(XR); pred=clf.classes_[prob.argmax(1)]
    top3=clf.classes_[np.argsort(-prob,axis=1)[:,:3]]
    present=np.unique(yR); pm=np.isin(pred,present)
    res[m]={"acc_all":accuracy_score(yR,pred),
            "acc_hard":accuracy_score(yR[masks["最难25%（LBE最高）"]],pred[masks["最难25%（LBE最高）"]]),
            "acc_sib":accuracy_score(yR[masks["父子同族(有兄弟类)"]],pred[masks["父子同族(有兄弟类)"]]),
            "macroF1":f1_score(yR,pred,average="macro",labels=present,zero_division=0),
            "top3":float(np.mean([yR[i] in top3[i] for i in range(len(yR))])),
            "nll":log_loss(yR,prob,labels=clf.classes_),
            "prob_true":float(np.mean([prob[i,list(clf.classes_).index(yR[i])] for i in range(len(yR))]))}
for k in ["acc_all","acc_hard","acc_sib","macroF1","top3","prob_true","nll"]:
    print(f"{k:26s}" + "".join(f"{res[m][k]:12.4f}" for m in ["ent","order","rand"]))
print("\n（nll 越低越好，其余越高越好；测试句 %d 条，其中难样本 %d 条、父子同族 %d 条）"%(len(yR),masks["最难25%（LBE最高）"].sum(),masks["父子同族(有兄弟类)"].sum()))
json.dump(res,open(BASE+"/results/entropy_metric_study.json","w"),ensure_ascii=False,indent=1)
