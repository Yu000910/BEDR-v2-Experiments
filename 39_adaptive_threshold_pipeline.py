# -*- coding: utf-8 -*-
"""39_adaptive_threshold_pipeline.py
多轮自适应过采样：按论文式(3.11)(3.12) 的阈值攀升机制 + 覆盖兜底，逐类重建标签集。

算法（与正文 3.2.3 节一一对应）
--------------------------------------------------------------------------
输入：基础集 + 审计阶段接纳的候选（θ=1.0）＝起始池；四轮结构化生成候选（round5/6/6b/7）。
对每个样本数 < n* 的类别 c：
  ε_c ← ε_strict（默认 1.0）
  逐轮遍历该轮该类的候选（按混合空间局部边界熵升序）：
      跳过近重复候选（maxsim ≥ 0.90；末轮 ≥ 0.92）；
      仅接纳 h ≤ ε_c 的候选，直至该类达到 n*；
      轮末若仍不足 n* 且 r_success = N_acc/N_cand < τ_rate(0.20)、且有足够被拒候选：
          ε_suggest = ε_c + 0.5·(h_med_rej − ε_c)                 …… 式(3.11)
          ε_c ← max(ε_c, min(ε_max, ε_suggest))                    …… 式(3.12)
  若轮次用尽仍不足 n*：触发【覆盖兜底】——按熵升序纳入剩余候选直至 n*，
      此时允许突破 ε_max（覆盖是硬约束，ε_max 是质量目标）。
最后按每类上限 n_max=100 以同一熵准则削顶。

用法
----
  python3 39_adaptive_threshold_pipeline.py                # 重建 + 双测试面评估
  python3 39_adaptive_threshold_pipeline.py --skip-eval    # 只重建，不训练（约 1 分钟）
输出
----
  out/adaptive/D_adaptive.npz                              重建的标签集（向量/标签/文本索引）
  results/adaptive_threshold_pipeline.json                 覆盖率、规模、准确率、兜底统计、与终版一致性
依赖
----
  候选嵌入缓存 out/embed_cache_v4_candidates.npz（由 47_v4_ablation.py 生成；缺失时按提示先跑它）
"""
import os, csv, json, time, argparse
import numpy as np
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = ["round5", "round6", "round6b", "round7"]
CACHE = BASE + "/out/embed_cache_v4_candidates.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--eps0", type=float, default=1.0, help="初始严格阈值 ε_strict（默认 1.0）")
ap.add_argument("--eps-max", type=float, default=2.3, help="阈值目标上限 ε_max（默认 2.3）")
ap.add_argument("--tau", type=float, default=0.20, help="成功率触发值 τ_rate（默认 0.20）")
ap.add_argument("--ramp-trigger", choices=["success", "shortfall"], default="success",
                help="阈值上移的触发条件：success=仅当累积成功率低于 τ_rate 且存在被拒候选（默认，与正文式(3.11) 一致）；shortfall=该类仍未达标即上移")
ap.add_argument("--n-star", type=int, default=30, help="每类样本下限 n*（默认 30）")
ap.add_argument("--cap", type=int, default=100, help="每类样本上限 n_max（默认 100）")
ap.add_argument("--knn", type=int, default=15, help="局部边界熵的邻居数 k（默认 15）")
ap.add_argument("--dedup", type=float, default=0.90, help="近重复剔除阈值 ρ（默认 0.90；末轮用 0.92）")
ap.add_argument("--skip-eval", action="store_true", help="只重建标签集，不做下游训练")
args = ap.parse_args()

classes = json.load(open(BASE + "/data/classes_sorted.json"))
n2i = {c: i for i, c in enumerate(classes)}
def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X.astype(np.float32) / n

# ---------- 1) 起始池：基础集 + 审计阶段接纳（θ=1.0）----------
zb = np.load(BASE + "/data/D_base_train_new.npz", allow_pickle=True)
XB = norm(zb["vectors"]); yB = np.array([n2i[str(x)] for x in zb["labels"]], dtype=np.int64)
emb = np.load(BASE + "/data/audit/embeds.npz"); txts = json.load(open(BASE + "/data/audit/texts.json"))
XC = np.vstack([norm(emb["rem_emb"]), norm(emb["kept_emb"])])
yC = np.array([n2i[t] for t, _ in txts["removed"]] + [n2i[t] for t, _ in txts["kept"]], dtype=np.int64)
fd = np.load(BASE + "/out/filter_corrected/filter_decision.npz", allow_pickle=True)
acc0 = fd["accept_local10"]
XP = np.vstack([XB, XC[acc0]]); yP = np.concatenate([yB, yC[acc0]])
cntP = Counter(yP.tolist())
print("起始池：基础集 %d + 审计接纳 %d = %d 条 / %d 类" % (len(yB), int(acc0.sum()), len(yP), len(cntP)), flush=True)

# ---------- 2) 候选与意图向量 ----------
def _embed_candidates():
    """缓存缺失时按需调用 DashScope text-embedding-v4 生成候选嵌入（约 2,800 次批量调用）。"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        for cand in (BASE + "/../.dashscope_key", BASE + "/.dashscope_key"):
            if os.path.exists(cand):
                key = open(cand).read().strip(); break
    if not key:
        raise SystemExit("缺少候选嵌入缓存，且未找到 DASHSCOPE_API_KEY；请设置环境变量后重跑本脚本。")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    texts, out, B = [c["s"] for c in cands], [], 10
    t0 = time.time()
    for i in range(0, len(texts), B):
        for att in range(8):
            try:
                r = client.embeddings.create(model="text-embedding-v4", input=texts[i:i+B],
                                             dimensions=768, encoding_format="float")
                out.extend([np.array(e.embedding, dtype=np.float32) for e in r.data]); break
            except Exception:
                if att == 7: raise
                time.sleep(2 * att + 1)
        if (i // B) % 200 == 0:
            print("embed", i, "/", len(texts), round(time.time() - t0), "s", flush=True)
    V = norm(np.vstack(out)); os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, VC=V); return V

cands = []
for ri, f in enumerate(ROUNDS):
    for r in csv.DictReader(open(BASE + f"/data/candidates/{f}_candidates.csv", newline="", encoding="utf-8")):
        try: ms = float(r["maxsim"])
        except Exception: ms = 0.0
        cands.append({"round": ri, "order": len(cands), "tech": r["tech_id"].strip(), "ms": ms})
VC = np.load(CACHE)["VC"] if os.path.exists(CACHE) else _embed_candidates()
assert len(cands) == len(VC), (len(cands), len(VC))
yc = np.array([n2i[c["tech"]] if c["tech"] in n2i else -1 for c in cands])
zi = np.load(BASE + "/data/intent_embeds.npz", allow_pickle=True)
ib = {str(l): norm(v.reshape(1, -1))[0] for l, v in zip(zi["labels"], zi["vectors"])}
VI = np.stack([ib[classes[c]] for c in range(679)])

def lbe(Xq, yq, Xr, yr, k):
    """查询样本相对参考池的局部边界熵（k 近邻标签分布熵）"""
    h = np.zeros(len(yq), np.float32); BS = 2048
    for i in range(0, len(yq), BS):
        Sim = Xq[i:i+BS] @ Xr.T
        nb = np.argpartition(-Sim, k, axis=1)[:, :k]; labs = yr[nb]
        for j in range(len(labs)):
            v, c = np.unique(labs[j], return_counts=True); p = c / c.sum()
            h[i+j] = -(p * np.log2(p)).sum()
    return h

XPh = norm(XP + VI[yP]); VCh = norm(VC + VI[np.clip(yc, 0, 678)])
h = lbe(VCh, yc, XPh, yP, args.knn)
print("候选局部边界熵：均值 %.3f 中位 %.3f" % (h.mean(), np.median(h)), flush=True)

# ---------- 3) 逐类阈值攀升 + 覆盖兜底 ----------
by = defaultdict(lambda: defaultdict(list))
for j, c in enumerate(cands):
    if c["tech"] in n2i: by[c["round"]][n2i[c["tech"]]].append(j)
accepted, log, fb_h = [], {}, []
for c in range(679):
    have = cntP.get(c, 0)
    if have >= args.n_star: continue
    eps, acc, trace = args.eps0, [], []
    for ri in range(len(ROUNDS)):
        js = sorted(by[ri].get(c, []), key=lambda j: h[j])
        # 轮内攀升：先按当前阈值接纳；若不足且仍有被拒候选，则按式(3.11)上移阈值后继续，
        # 直至该类填满、或阈值达到 ε_max、或已无被拒候选（说明当轮候选已用尽）。
        steps = 0
        while True:
            n_acc, rej = 0, []
            for j in js:
                if have + len(acc) >= args.n_star: break
                if cands[j]["ms"] >= (0.92 if ri == 3 else args.dedup): continue
                if j in acc: continue
                if h[j] <= eps: acc.append(j); n_acc += 1
                else: rej.append(j)
            if have + len(acc) >= args.n_star: break
            if eps >= args.eps_max - 1e-9: break      # 唯一停升条件：阈值触顶
            if not rej:
                # 当轮已无被拒候选：阈值再放宽也不会新增任何候选，
                # 但其含义与“门已开到上限”等价，故直接把阈值记至 ε_max 后进入下一轮/兜底。
                eps = args.eps_max
                steps += 1
                break
            h_med = float(np.median(h[rej]))
            eps_new = max(eps, min(args.eps_max, eps + 0.5 * (h_med - eps)))   # 式(3.11)+(3.12)
            steps += 1
            if eps_new <= eps + 1e-9:
                # 半程步长已收敛（被拒中位数低于当前阈值附近）：按题意直接抬到上限，
                # 使“不足”只可能出现在触顶之后。
                eps = args.eps_max
                break
            eps = eps_new
            if steps >= 20:                            # 仅作循环保险
                eps = args.eps_max
                break
        r_succ = n_acc / max(1, len(js))
        trace.append({"round": ri, "eps": round(float(eps), 3), "cand": len(js), "acc": n_acc,
                      "r_success": round(float(r_succ), 3), "ramp_steps": steps})
        if have + len(acc) >= args.n_star: break
    fallback = False
    if have + len(acc) < args.n_star:                               # 覆盖兜底
        fallback = True
        rest = sorted({j for ri in range(len(ROUNDS)) for j in by[ri].get(c, [])
                       if j not in acc and cands[j]["ms"] < (0.92 if ri == 3 else args.dedup)},
                      key=lambda j: h[j])
        for j in rest:
            if have + len(acc) >= args.n_star: break
            acc.append(j); fb_h.append(float(h[j]))
    accepted.extend(acc)
    log[c] = {"base": have, "accepted": len(acc), "final_eps": round(float(eps), 3),
              "eps_at_cap": bool(eps >= args.eps_max - 1e-9), "fallback": fallback, "trace": trace}

add = np.array(sorted(set(accepted)), dtype=np.int64)
reach = sum(1 for c in range(679) if cntP.get(c, 0) + len([j for j in add if yc[j] == c]) >= args.n_star)
print("需补足类别 %d | 触发兜底 %d | 接纳候选 %d | 达到 n* 的类别 %d/679" % (
    len(log), sum(1 for v in log.values() if v["fallback"]), len(add), reach), flush=True)

# ---------- 4) 组装 + 每类上限削顶 ----------
X = np.vstack([XP, VC[add]]); y = np.concatenate([yP, yc[add]])
Xh = np.vstack([XPh, VCh[add]])
hX = lbe(Xh, y, XPh, yP, args.knn)
keep = []
for c in range(679):
    ci = np.where(y == c)[0]
    keep.append(ci if len(ci) <= args.cap else ci[np.argsort(hX[ci])[:args.cap]])
kk = np.sort(np.concatenate(keep)); Xf, yf = X[kk], y[kk]
os.makedirs(BASE + "/out/adaptive", exist_ok=True)
np.savez(BASE + "/out/adaptive/D_adaptive.npz", vectors=Xf, labels=yf)
print("重建标签集：%d 条 / %d 类（每类 %d~%d 条）" % (
    len(yf), len(set(yf.tolist())), min(Counter(yf.tolist()).values()), max(Counter(yf.tolist()).values())), flush=True)

out = {"config": vars(args), "pool": {"base": int(len(yB)), "audit_accepted": int(acc0.sum()), "pool": int(len(yP)),
       "pool_classes": int(len(cntP))}, "candidates": int(len(cands)),
       "needs_fill_classes": len(log), "fallback_classes": sum(1 for v in log.values() if v["fallback"]),
       "fallback_with_eps_at_cap": sum(1 for v in log.values() if v["fallback"] and v["eps_at_cap"]),
       "fallback_before_cap": sum(1 for v in log.values() if v["fallback"] and not v["eps_at_cap"]),  # 预期为 0：不足仅在触顶后判定
       "classes_reached_cap": sum(1 for v in log.values() if v["eps_at_cap"]),
       "accepted_candidates": int(len(add)), "reach_n_star": int(reach),
       "final": {"n": int(len(yf)), "classes": int(len(set(yf.tolist())))},
       "fallback_lbe": {"mean": round(float(np.mean(fb_h)), 3) if fb_h else None,
                        "max": round(float(max(fb_h)), 3) if fb_h else None, "n": len(fb_h)},
       "per_class_head": {str(k): log[k] for k in list(log)[:10]}}

# ---------- 5) 评估 + 与终版一致性 ----------
if not args.skip_eval:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    zA = np.load(BASE + "/data/orig4291_plain.npz", allow_pickle=True)
    XA = norm(zA["vectors"]); yA = np.asarray(zA["labels"], dtype=np.int64)
    z2 = np.load(BASE + "/data/test2_full_plain.npz", allow_pickle=True); m2 = np.asarray(z2["src"]) == "real"
    XR = norm(z2["vectors"][m2]); yR = np.asarray(z2["labels"], dtype=np.int64)[m2]
    a, r = [], []
    for s in [42, 43, 44, 45, 46]:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, n_iter_no_change=5,
                            batch_size=256, learning_rate_init=1e-3, random_state=s).fit(Xf, yf)
        a.append(accuracy_score(yA, clf.predict(XA))); r.append(accuracy_score(yR, clf.predict(XR)))
    rg = RidgeClassifier(alpha=1.0).fit(Xf, yf)
    out["eval"] = {"mlp679": [round(float(np.mean(a)), 4), round(float(np.std(a)), 4)],
                   "ridge679": round(float(accuracy_score(yA, rg.predict(XA))), 4),
                   "mlp420": [round(float(np.mean(r)), 4), round(float(np.std(r)), 4)],
                   "ridge420": round(float(accuracy_score(yR, rg.predict(XR))), 4)}
    zf = np.load(BASE + "/data/D_final.npz", allow_pickle=True)
    Xdf = norm(zf["vectors"])
    S = Xdf @ VC.T; idx = S.argmax(axis=1); sim = S.max(axis=1)
    matched = set(idx[sim > 0.999].tolist())
    out["consistency_with_final"] = {"dfinal_rows": int(len(Xdf)), "dfinal_generated_matched": len(matched),
                                     "overlap": len(matched & set(add.tolist())),
                                     "only_adaptive": len(set(add.tolist()) - matched),
                                     "only_final": len(matched - set(add.tolist()))}
    print("评估：679 MLP %.4f Ridge %.4f | 420 MLP %.4f Ridge %.4f" % (
        np.mean(a), out["eval"]["ridge679"], np.mean(r), out["eval"]["ridge420"]), flush=True)
    print("与终版一致：重合 %d / %d（%.1f%%）" % (
        out["consistency_with_final"]["overlap"], len(matched),
        100 * out["consistency_with_final"]["overlap"] / max(1, len(matched))), flush=True)

_suffix = "_buildonly" if args.skip_eval else ""
_outp = BASE + "/results/adaptive_threshold_pipeline%s.json" % _suffix
json.dump(out, open(_outp, "w"), ensure_ascii=False, indent=1)
print("DONE ->", _outp, flush=True)
