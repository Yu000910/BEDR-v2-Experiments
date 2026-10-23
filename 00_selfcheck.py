# -*- coding: utf-8 -*-
"""00_selfcheck.py —— 定稿包自检：确认数据、模型、结果齐全且尺寸正确。
用法：cd 02_code && python 00_selfcheck.py
"""
import os, json
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    ("data/D_base_train_new.npz", "基础集"), ("data/D_naive_full.npz", "朴素集"),
    ("data/D_arms_HYBRID.npz", "过采样后"), ("data/D_final.npz", "终版标签集"),
    ("data/orig4291_plain.npz", "全覆盖测试集"), ("data/test2_full_plain.npz", "真实语句测试集"),
    ("data/intent_embeds.npz", "意图向量"), ("data/classes_sorted.json", "类别映射"),
    ("data/audit/embeds.npz", "审计嵌入"), ("data/audit/texts.json", "审计文本"),
    ("data/filter_decision.npz", "筛选决策"),
    ("models/ridge.joblib", "Ridge"), ("models/mlp_seed42.joblib", "MLP seed42"),
    ("results/six_metrics.json", "T1546 拓扑"), ("results/gbe_four_models.json", "四模型 GBE"),
    ("results/adaptive_threshold_pipeline.json", "自适应阈值实现"),
    ("results/ablation_v4_fair.json", "熵筛选等规模"), ("results/entropy_smallbudget.json", "熵筛选紧预算"),
    ("results/rq5_rerun.json", "外部基线"), ("results/sensitivity.json", "敏感性"),
]
ok = True
for rel, name in FILES:
    p = os.path.join(BASE, rel)
    exists = os.path.exists(p)
    ok &= exists
    print(f"[{'OK' if exists else 'MISSING'}] {rel:<48s} {name}")
for npz in ["data/D_base_train_new.npz", "data/D_naive_full.npz", "data/D_final.npz",
            "data/orig4291_plain.npz", "data/test2_full_plain.npz"]:
    z = np.load(os.path.join(BASE, npz), allow_pickle=True)
    y = z["labels"]
    print(f"      {npz:<34s} vectors={z['vectors'].shape} labels={len(y)}")
d = json.load(open(os.path.join(BASE, "results/gbe_four_models.json")))
print("\n四模型 GBE:", {k: d[k]["gbe"] for k in ["B", "C", "Cp", "A"]})
print("自检", "通过 ✅" if ok else "失败 ❌")
