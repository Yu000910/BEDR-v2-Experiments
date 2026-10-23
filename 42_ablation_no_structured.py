# -*- coding: utf-8 -*-
"""42_ablation_no_structured.py
消融臂①：移除结构化语义解耦（降级为"朴素提示词生成 + 普通空间熵筛选"）。

设计说明（与论文 3.3.3(4) 一致）：
    去掉结构化语义解耦后，官方意图无法再从技术描述中提取，混合向量参考系随之不可构建，
    熵筛选只能退回普通向量空间。该臂因此反映"组件 + 其依赖链"一并移除的整体效果。
    为与完整流水线可比，每类条数按"熵升序"削至优化集（D_final）的同类条数，
    最终规模 23,562 条（= 逐类 min(朴素集条数, 优化集条数) 之和）。
输入：data/D_naive_full.npz、D_final.npz、D_base_train_new.npz、audit/、两个测试面
输出：results/ablation_no_structured.json
"""
import os, json
import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
B = os.path.dirname(os.path.abspath(__file__))   # 包根目录
def norm(X):
    n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1.0; return X.astype(np.float32)/n
n2i={c:i for i,c in enumerate(json.load(open(B+"/data/classes_sorted.json")))}
zB=np.load(B+"/data/D_base_train_new.npz",allow_pickle=True); XB=norm(zB["vectors"]); yB=np.array([n2i[str(t)] for t in zB["labels"]],dtype=np.int64)
emb=np.load(B+"/data/audit/embeds.npz"); txts=json.load(open(B+"/data/audit/texts.json"))
XC=np.vstack([norm(emb["rem_emb"]),norm(emb["kept_emb"])]); yC=np.array([n2i[t] for t,_ in txts["removed"]]+[n2i[t] for t,_ in txts["kept"]],dtype=np.int64)
fd=np.load(B+"/data/filter_decision.npz",allow_pickle=True)
XP=np.vstack([XB,XC[fd["accept_local10"]]]); yP=np.concatenate([yB,yC[fd["accept_local10"]]])
zN=np.load(B+"/data/D_naive_full.npz",allow_pickle=True); XN=norm(zN["vectors"]); yN=np.array([n2i[str(t)] for t in zN["labels"]],dtype=np.int64)
zE=np.load(B+"/data/D_final.npz",allow_pickle=True); yE=np.asarray(zE["labels"],dtype=np.int64)
cntE=Counter(yE.tolist())
def lbe(Xq,yq,k=15,BS=2048):
    h=np.zeros(len(yq),np.float32)
    for i in range(0,len(q:=yq),BS):
        D=1.0-Xq[i:i+BS]@XP.T          # 查询集与参考池不同，无需自排除
        nb=np.argpartition(D,k-1,axis=1)[:,:k]; labs=yP[nb]
        for j in range(len(labs)):
            v,c=np.unique(labs[j],return_counts=True); p=c/c.sum(); h[i+j]=-(p*np.log2(p)).sum()
    return h
print("朴素集普通空间 LBE 计算中…",flush=True)
hN=lbe(XN,yN)
keep=[]
for c in range(679):
    ci=np.where(yN==c)[0]
    if len(ci)==0: continue
    need=cntE.get(c,0)
    keep.append(ci[np.argsort(hN[ci])[:need]] if len(ci)>need else ci)
kk=np.sort(np.concatenate(keep)); X1,y1=XN[kk],yN[kk]
print("①-arm(朴素提示词+普通空间熵筛选+等规模): %d 条 / %d 类"%(len(y1),len(set(y1.tolist()))),flush=True)
zA=np.load(B+"/data/orig4291_plain.npz",allow_pickle=True); XA=norm(zA["vectors"]); yA=np.asarray(zA["labels"],dtype=np.int64)
z2=np.load(B+"/data/test2_full_plain.npz",allow_pickle=True); m=np.asarray(z2["src"])=="real"
XR=norm(z2["vectors"][m]); yR=np.asarray(z2["labels"],dtype=np.int64)[m]
a=[];r=[]
for s in [42,43,44,45,46]:
    clf=MLPClassifier(hidden_layer_sizes=(256,),max_iter=20,early_stopping=True,n_iter_no_change=5,batch_size=256,learning_rate_init=1e-3,random_state=s).fit(X1,y1)
    a.append(accuracy_score(yA,clf.predict(XA))); r.append(accuracy_score(yR,clf.predict(XR)))
rg=RidgeClassifier(alpha=1.0).fit(X1,y1)
res={"n":int(len(y1)),"mlp679":[round(float(np.mean(a)),4),round(float(np.std(a)),4)],
     "ridge679":round(float(accuracy_score(yA,rg.predict(XA))),4),
     "mlp420":[round(float(np.mean(r)),4),round(float(np.std(r)),4)],
     "ridge420":round(float(accuracy_score(yR,rg.predict(XR))),4)}
print("结果:",res,flush=True)
json.dump(res,open(B+"/results/ablation_no_structured.json","w"),ensure_ascii=False,indent=1)
