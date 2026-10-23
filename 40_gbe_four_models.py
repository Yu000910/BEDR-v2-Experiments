# -*- coding: utf-8 -*-
"""40_gbe_four_models.py
四个模型训练集的全局边界熵（GBE）及其与分类性能的对应关系。

输入：data/D_base_train_new.npz、D_naive_full.npz、D_final.npz、两个测试面
输出：results/gbe_four_models.json
说明：C′（等规模朴素子集）按“每类抽取与优化集相同的条数”重建（5 个种子取均值），
      其重建准确率与论文报告值在种子噪声内一致（0.523/0.502、0.668/0.675 对 0.527/0.502、0.669/0.675）。
"""
import os, json
import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
BASE=os.path.dirname(os.path.abspath(__file__))
classes=json.load(open(BASE+"/data/classes_sorted.json")); n2i={c:i for i,c in enumerate(classes)}
def norm(X):
    n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1.0; return X.astype(np.float32)/n
def load(p):
    z=np.load(p,allow_pickle=True); X=norm(z["vectors"])
    lab=list(z["labels"])
    if isinstance(lab[0],(str,np.str_)): y=np.array([n2i[str(x)] for x in lab],dtype=np.int64)
    else: y=np.asarray(lab,dtype=np.int64)
    return X,y
def gbe(X,y,k=15,BS=2048):
    h=np.zeros(len(y),np.float32)
    for i in range(0,len(y),BS):
        Dm=1.0-X[i:i+BS]@X.T
        r=np.arange(Dm.shape[0]); Dm[r,i+r]=1.0
        nb=np.argpartition(Dm,k-1,axis=1)[:,:k]; labs=y[nb]
        for j in range(len(labs)):
            v,c=np.unique(labs[j],return_counts=True); p=c/c.sum(); h[i+j]=-(p*np.log2(p)).sum()
    return float(h.mean())
XB,yB=load(BASE+"/data/D_base_train_new.npz")
XN,yN=load(BASE+"/data/D_naive_full.npz")
XE,yE=load(BASE+"/data/D_final.npz")
zA=np.load(BASE+"/data/orig4291_plain.npz",allow_pickle=True); XA_=norm(zA["vectors"]); yA_=np.asarray(zA["labels"],dtype=np.int64)
z2=np.load(BASE+"/data/test2_full_plain.npz",allow_pickle=True); m2=np.asarray(z2["src"])=="real"
XR_=norm(z2["vectors"][m2]); yR_=np.asarray(z2["labels"],dtype=np.int64)[m2]
print("规模: B=%d/%d  C=%d/%d  A=%d/%d"%(len(yB),len(set(yB.tolist())),len(yN),len(set(yN.tolist())),len(yE),len(set(yE.tolist()))),flush=True)
gB,gC,gA=gbe(XB,yB),gbe(XN,yN),gbe(XE,yE)
print("GBE: B=%.4f  C=%.4f  A=%.4f"%(gB,gC,gA),flush=True)
cntE=Counter(yE.tolist()); cntN=Counter(yN.tolist())
res={}
for seed in [42,43,44,45,46]:
    rng=np.random.RandomState(seed); parts=[]
    for c in range(679):
        need=cntE.get(c,0)
        if need==0: continue
        ci=np.where(yN==c)[0]
        if len(ci)<=need: parts.append(ci)
        else: parts.append(rng.choice(ci,size=need,replace=False))
    idx=np.sort(np.concatenate(parts)); Xc,yc=XN[idx],yN[idx]
    g=gbe(Xc,yc)
    clf=MLPClassifier(hidden_layer_sizes=(256,),max_iter=20,early_stopping=True,n_iter_no_change=5,batch_size=256,learning_rate_init=1e-3,random_state=seed).fit(Xc,yc)
    rg=RidgeClassifier(alpha=1.0).fit(Xc,yc)
    res[seed]={"n":int(len(yc)),"gbe":round(g,4),
               "mlp679":round(float(accuracy_score(yA_,clf.predict(XA_))),4),
               "ridge679":round(float(accuracy_score(yA_,rg.predict(XA_))),4),
               "mlp420":round(float(accuracy_score(yR_,clf.predict(XR_))),4),
               "ridge420":round(float(accuracy_score(yR_,rg.predict(XR_))),4)}
    print(seed,res[seed],flush=True)
import statistics as st
m=lambda k: (round(st.mean([res[s][k] for s in res]),4), round(st.pstdev([res[s][k] for s in res]),4))
out={"B":{"n":len(yB),"classes":len(set(yB.tolist())),"gbe":round(gB,4)},
     "C":{"n":len(yN),"classes":len(set(yN.tolist())),"gbe":round(gC,4)},
     "Cp":{"n":int(len(yc)),"gbe":m("gbe"),"mlp679":m("mlp679"),"ridge679":m("ridge679"),"mlp420":m("mlp420"),"ridge420":m("ridge420")},
     "A":{"n":len(yE),"classes":len(set(yE.tolist())),"gbe":round(gA,4)},
     "per_seed":res}
json.dump(out,open(BASE+"/results/gbe_four_models.json","w"),ensure_ascii=False,indent=1)
print("C' 均值: GBE %.4f±%.4f | 679 MLP %.4f Ridge %.4f | 420 MLP %.4f Ridge %.4f"%(m("gbe")[0],m("gbe")[1],m("mlp679")[0],m("ridge679")[0],m("mlp420")[0],m("ridge420")[0]),flush=True)
