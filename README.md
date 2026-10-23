# BEDR-v2 实验复现包

本目录包含小论文《BEDR：面向 ATT&CK 语义解耦的边界熵驱动数据增强》（修改版）的全部实验代码。

**核心结论**：BEDR-v2（结构化语义解耦生成 + 边界熵双向调控 + 动态诊断-补录循环 + 去重）构建的 679 类标签集，在可部署协议（plain 空间、真实报告测试集）下，以少于朴素 LLM 增强基线 29% 的样本量，在 MLP 与 Ridge 两个分类器上均显著优于朴素基线。

---

## 目录结构

编号即执行顺序：

    01_eval_judge_swap.py        等量对齐 x 六裁判对决（数量/质量解耦）
    02_eval_diversity.py         类内多样性 + 真实行覆盖度测量
    03_eval_tail_flicker.py      尾类闪烁验证（10 种子 + 朴素-朴素对照）
    04_audit_fidelity.py         保真审计（被删/保留/朴素 vs 官方 STIX 定义）
    05_eval_filter_corrected.py  过滤修正对照（全局边际判据 / plain 局部熵 / hybrid 局部熵）
    06_eval_saturation.py        样本数-准确率饱和曲线（30/类的定量依据）
    07_gen_fix_pilot.py          修复试点①（多样性指令版，负面结果）
    08_eval_fix_pilot.py         修复试点①评估
    09_gen_fix2_pilot.py         修复试点②（风格范本版，负面结果：名人效应）
    10_eval_fix2.py              修复试点②评估
    11_gen_round5_v2.py          第 5 轮：294 个尾类三臂候选生成（plain/hybrid 双空间熵）
    12_build_round5_arms.py      第 5 轮三臂组装（NONE/PLAIN/HYBRID）+ 嵌入
    13_eval_bedrv2.py            第 5 轮五方对决（三臂 vs 朴素 vs 原版）
    14_gen_round6_v2.py          第 6 轮：190 类候选生成
    15_gen_round6b_v2.py         第 6b 轮：53 个空零样本类候选生成
    16_build_round6_arms.py      第 6 轮组装 + 679 类完整数据集导出
    17_fill_underfilled.py       放宽阶梯补缺（严格阈值填不满的类）
    18_gen_round7.py             第 7 轮：48 个欠类定向补录（去重阈值 0.92）
    19_eval_final_matrix2.py     终验矩阵（全量/等量 MLP + Ridge，双测试面，5 种子，t 检验）
    20_eval_head_undersample.py   头部熵引导欠采样实验（保留最低熵 80/100 条对照）
    21_eval_top100.py             终版数据集验证（TOP100 + 5 种子 + t 检验 + 尾类）

辅助目录：
    data/      全部数据集与标签映射（最终数据集 = data/D_bedrv2_HYBRID_top100.npz）
    models/    训练好的模型权重（MLP 5 种子 + Ridge，joblib 格式）
    results/   全部实验输出 JSON（终验矩阵、t 检验、饱和曲线、保真审计等）

## 环境

    Python 3.12（Anaconda）
    pip install numpy scipy scikit-learn openai

## 数据（data/ 托管于 Hugging Face）

data/ 目录（约 780 MB）包含全部数据集，托管于 Hugging Face：

    https://huggingface.co/datasets/Andou2yu/BEDR-v2-Experiments-Data

下载并放置到仓库 data/ 目录：

    pip install huggingface_hub
    huggingface-cli download Andou2yu/BEDR-v2-Experiments-Data --repo-type dataset --local-dir data

data/ 内含：基线集、朴素基线集、原论文 BEDR 集、BEDR-v2 三臂（NONE/PLAIN/HYBRID）、终版数据集（D_bedrv2_HYBRID_top100.npz，24,174 条，679 类，每类 30~100 条）、两个测试面、类映射与零样本名单、每类官方意图嵌入、ATT&CK 官方定义（enterprise-attack.json）与意图/动作链/实体图（intent_action_entity.json）。

## API Keys（脚本内为占位符，需替换）

生成/嵌入类脚本（04/07/09/11/12/14/15/16/17/18）需要两个 API，脚本内密钥已替换为占位符，运行前请替换为你自己的密钥：

- DeepSeek chat（样本生成）：将脚本中的 YOUR_DEEPSEEK_API_KEY 替换为你的 DeepSeek API Key
- DashScope text-embedding-v4（768 维嵌入）：将脚本中的 YOUR_DASHSCOPE_API_KEY 替换为你的 DashScope API Key

## 路径说明

所有脚本的 BASE 均指向脚本所在目录（os.path.dirname(os.path.abspath(__file__))），clone 后无需修改路径即可运行；输出写入仓库内 out/ 与 results/ 目录（脚本自动创建）。

## 执行顺序（六阶段）

阶段 0：数据准备（原仓库脚本构建上述 npz 与映射文件），本包假设已存在。

阶段 1：诊断（01-03，纯计算，无需 API）
    01 等量对齐六裁判：输出 out/judge_swap/
    02 多样性/覆盖度：输出 out/diversity/
    03 尾类验证：输出 out/tail_flicker/

阶段 2：保真审计（04，需嵌入 API，约 30 分钟）
    输出 out/fidelity_audit/（含 679 条官方定义嵌入、被删/保留/朴素样本嵌入）

阶段 3：过滤修正对照（05，纯计算）
    输出 out/filter_corrected/（全局边际判据与两空间局部熵的接受决策）

阶段 4：数据集构建（06-18）
    06 饱和曲线 → 定每类目标 30
    11-12 第 5 轮三臂 → D_round5_{NONE,PLAIN,HYBRID}.npz
    13 五方对决 → judge_results.json
    14-16 第 6 轮 + 组装 → D_bedrv2_{NONE,PLAIN,HYBRID}.npz（679 类完整）
    17-18 补缺 + 第 7 轮 → HYBRID 臂每类 ≥30
    最终数据集：out/round5/D_bedrv2_HYBRID.npz（26,490 条，679 类）

阶段 5：终验矩阵（19，约 40 分钟）
    输出 out/final_matrix/（final_matrix.json、ttests.json、tail.json）

## 核心结果速查

终版数据集 = data/D_bedrv2_HYBRID_top100.npz（24,174 条，679 类，每类 30~100 条；头部类保留 hybrid 熵最低的 100 条）

终验矩阵（真实行测试面 B，2,984 条；5 种子均值 + 配对 t 检验）：

    配置      TOP100      朴素      原版
    MLP       0.712       0.690     0.613
    Ridge     0.686       0.661     0.663
    显著性：MLP +2.2 点 p=0.003；Ridge +2.5 点（确定性）

原测试面（表面 A，4,291 条，679 类）：
    MLP 0.676 vs 0.556（+12.0 点，p=5.0e-06）；Ridge 0.605 vs 0.525（+8.0 点）

欠采样对照（MLP / Ridge，表面 B 全量）：
    不欠采样 26,490 条：0.731 / 0.627
    保留 100 条 24,174 条：0.712 / 0.686
    保留 80 条 23,562 条：0.689 / 0.690

尾类子集（294 类，694 行）：
    - 等量预算下 BEDR-v2 反超朴素 6-8 点（NC/Ridge）
    - 全量 MLP：TOP100 0.543 vs 朴素 0.553（欠采样只动头部类；尾类差距主要在等量口径下反转）

饱和曲线（朴素，每类 5→50 条）：
    5→10 +0.088；10→15 +0.037；15→20 +0.017；30→40 +0.014
    → 拐点 15-20，每类目标定 30 的定量依据

保真审计（官方 STIX 定义余弦）：
    结构化保留 0.631 vs 朴素 0.616（p<0.0001）
    被熵过滤删除 0.626 vs 保留 0.631（无显著差异 → 过滤对质量中性）

边界熵反差（协议审计证据）：
    hybrid 空间平均局部熵 0.50 vs plain 空间 2.36（4.7 倍）
    → 锚点注入使边界"人为干净"，该反差可作为嵌入协议审计指标

零样本类（105 类）：
    BEDR-v2 覆盖 105/105；类心-定义对齐 0.752 vs 朴素 0.778（如实披露朴素略优）

## 协议说明（重要）

1. 训练/测试严格零重叠；测试 = 真实威胁报告原句（表面 B）或原论文测试集（表面 A，含零样本类合成行）。
2. 可部署协议 = plain 向量训练与测试；hybrid 向量（意图锚点+句子平均）仅用于生成期过滤（意图贴合度信号），不参与训练/测试。
3. 零样本类无真实测试行，仅按官方定义口径评估（原型-定义对齐）。
4. 每类样本目标 30 由饱和曲线确定；补录为动态诊断-补录循环，轮数非固定。
5. 边界熵的角色：生成期选择信号 + 协议审计/诊断工具；不作为"质量筛"声称（保真审计与下游实验均不支持该声称）。

## 模型权重（models/）

    mlp_seed42.joblib ~ mlp_seed46.joblib   终版数据集上训练的 MLP 分类器（5 种子，sklearn MLPClassifier，256 隐藏层）
    ridge.joblib                           终版数据集上训练的 Ridge 分类器（确定性）

加载示例：import joblib; clf = joblib.load('models/mlp_seed42.joblib')

注：旧协议（hybrid 训练/测试）的 PyTorch 模型权重不在本仓库（该协议已被可部署 plain 协议取代，其权重仅具历史意义）。

## 已知负面结果（保留为消融证据）

- 修复试点①（多样性指令）：下游 -20 点（换词不换骨）
- 修复试点②（风格范本 + 名人效应）：下游 -13 至 -20 点（Mimikatz/Cobalt Strike 等明星工具名污染嵌入）
- 全局边际熵判据（理论原样实现）：78.5% 候选边际效应为负，无区分力
- hybrid 空间过滤（原论文实现）：全部 9 格下游最差（换 plain 空间回升最多 6 点）

## 指标复现对照表（论文中每个实验指标 <-> 脚本 <-> 输出）

| 论文指标 | 复现脚本 | 输出 |
| --- | --- | --- |
| 终验矩阵：真实面 MLP 0.712 / Ridge 0.686 与 t 检验 | 21_eval_top100.py | results/top100_matrix.json |
| 原测试面 MLP 提升 12.0 个百分点 | 21_eval_top100.py | results/top100_matrix.json |
| 五方对决矩阵与显著性（终版前身） | 19_eval_final_matrix2.py | results/final_matrix.json、ttests.json、tail.json |
| 尾类等预算高出 3~8 点（NC/Ridge，k=10/20/30） | 22_eval_tail_final.py | 运行日志（每档 BEDR vs NAIVE） |
| 官方定义语义对齐 0.631 vs 0.616 | 04_audit_fidelity.py | results/results.json |
| 边界熵反差 4.7 倍（hybrid 0.50 vs plain 2.36） | 05_eval_filter_corrected.py 的池熵计算 | 运行日志 |
| 锚点强度-熵单调曲线（受控实验 2.57→0.19） | 23_anchor_sweep.py | round5/anchor_sweep.json |
| 饱和曲线（每类 30 条的定量依据） | 06_eval_saturation.py | results/saturation.json |
| 头部欠采样对照（0.731 / 0.712 / 0.689） | 20_eval_head_undersample.py | 运行日志 |
| 三臂消融（无过滤 / plain 熵 / hybrid 熵） | 11_gen_round5_v2.py + 12_build_round5_arms.py + 13_eval_bedrv2.py | results/judge_results.json |
| 类内多样性与真实行覆盖度 | 02_eval_diversity.py | 运行日志 |
| 尾类闪烁验证（10 种子 + 对照） | 03_eval_tail_flicker.py | 运行日志 |
| 失败模式消融（多样性指令 / 名人效应） | 07~10 | 运行日志 |
| 数量-质量解耦（等量对齐六裁判） | 01_eval_judge_swap.py | 运行日志 |
| 全局边际判据对照（理论原样实现） | 05_eval_filter_corrected.py | round5/filter_decision.npz |

## 最终数据集与模型（论文主结果直接使用）

- 训练集：data/D_bedrv2_HYBRID_top100.npz（24,174 条，679 类，每类 30~100 条）
- 朴素基线：data/D_naive_full.npz；原论文 BEDR：data/D_bedr_train.npz
- 真实行测试面：data/test2_full_plain.npz（src 列标记 real）；原测试面：data/orig4291_plain.npz
- 模型权重：models/mlp_seed42~46.joblib（在终版数据集上训练）、models/ridge.joblib


## 补充复现脚本（24-34，与论文各表一一对应）

| 论文指标 | 脚本 | 输出 |
| --- | --- | --- |
| RQ5 外部基线全量口径（0.619/0.372 等） | 24_rq5_rerun.py | results/rq5_rerun.json |
| RQ5 外部基线等样本口径与 t 检验（0.356, p<0.001） | 25_rq5_equal5.py | results/rq5_equalbudget5.json |
| RQ8 多家族等量拓扑（轮廓 5/6、分离比 4/6） | 26_multifamily.py | results/multifamily_matched.json |
| RQ1 六项拓扑指标（T1546） | 27_six_metrics.py | results/six_metrics.json |
| RQ1 JS 散度（0.015/0.265/0.204） | 28_js_div.py | results/js_div.json |
| RQ7 零样本对齐配对检验（0.752 vs 0.778, t=-5.28） | 29_zero_align.py | results/zero_align.json |
| RQ2 三方等量口径（公共类集 k=5/10/30） | 30_common_matched.py | results/common_matched.json |
| RQ2 基线矩阵（0.704/0.638） | 31_baseline_matrix.py | results/baseline_matrix.json |
| RQ4 补录消融（第 5 轮后数据集矩阵） | 32_r5_matrix.py | results/r5_matrix.json |
| 终版数据集文本重建 | 33_rebuild_texts.py | 依赖 data/candidates/*.csv 与 data/audit/* |
| 熵反差 0.47 vs 2.30 + 锚点扫描 | 34_entropy_contrast.py | results/entropy_contrast.json |
| 参数敏感性（k 与阈值网格，3 种子） | 35_sensitivity.py | results/sensitivity.json |
| RQ1 Gini/JS 分布对齐表 | 36_gini.py | results/gini_js.json |

## data/ 子目录说明

- data/candidates/：第 5/6/6b/7 轮全部候选句 CSV（补录管线的原始输入，共约 3.2 万条）
- data/audit/：保真审计的嵌入与文本（官方定义嵌入、被删/保留/朴素样本嵌入）
- data/filter_decision.npz：plain 局部熵筛选决策（F_LOC10 池的构成依据）

---

## 最终实验脚本（37–48）与自检

在上述主流程之后，补充以下脚本用于论文最终报告的全部对照与消融实验（编号即论文中的呈现顺序，均以仓库根目录为工作目录，数据在 `data/`，结果写入 `results/`）：

    00_selfcheck.py                     自检：核对 data/models/results 是否齐全并打印四模型 GBE
    37_gbe_error_analysis.py            全局边界熵与下游错误率的关系
    38_oversampling_ablation.py         过采样各轮次的消融
    39_adaptive_threshold_pipeline.py   自适应接纳阈值（诊断-补录）流程的完整复现
    40_gbe_four_models.py               四种训练集（B / C / C' / A）的 GBE 对比
    41_tail_budget.py                   322 个长尾类别、等每类预算下的对比
    42_ablation_no_structured.py         消融①：去结构化语义解耦（降级为朴素提示词 + 普通空间筛选，23,562 条）
    43_ablation_entropy_fair.py         消融：等规模下熵筛选 vs 随机选择
    44_ablation_local_space.py          消融②：在普通空间计算熵（去意图锚定空间）
    45_entropy_ablation_bysubset.py     熵筛选在不同子集上的分层检验
    46_entropy_metrics.py               熵度量的替代方案比较
    47_v4_ablation.py                   等规模三臂消融汇总
    48_entropy_smallbudget.py           紧预算（每类 10 条 / 5 条）下的熵筛选增益

## 数据与结果

- `data/`：类别索引、可行子集、零样本类、结构化攻击图、真实语句测试集、筛选决策、意图向量等小文件；
- `results/`：论文每张表格与消融实验对应的 JSON 结果；
- `models/`：训练好的分类器（5 个 MLP 随机种子 + Ridge）。

体积较大的中间数组（`D_base_train_new.npz`、`D_naive_full.npz`、`D_arms_HYBRID.npz`、`D_final.npz`、`orig4291_plain.npz`、`test2_full_plain.npz`、`audit/embeds.npz`、`audit/texts.json`，合计约 450 MB）作为本仓库 **v1.0-artifacts** Release 的附件提供，下载后放入 `data/`（audit 两个文件放入 `data/audit/`）即可运行全部脚本。

## 复现入口

```bash
python 00_selfcheck.py      # 应输出：自检 通过，以及四模型 GBE {B: 3.0970, C: 2.5064, C': 2.7017, A: 2.2857}
```
