# BEDR: Boundary Entropy-Driven Resampling for Fine-Grained ATT&CK Technique Mapping

Experiment code, trained classifiers and result files for the paper

> **BEDR: Boundary Entropy-Driven Resampling with Large Language Models for Fine-Grained Cyber Threat Intelligence Mapping**

BEDR constructs a boundary-clear ATT&CK technique label set (679 techniques, 24,174 samples, 30–100 per class). The framework combines structured semantic decomposition of official technique descriptions, an intent-anchored hybrid vector space, local/global boundary entropy as an explicit quality objective, an adaptive diagnosis–refill loop, and entropy-guided selective undersampling. The label set is evaluated on a full-coverage test set (4,291 rows, 679 classes) and a real-sentence test set (2,984 rows, 420 classes) with two classifiers, size-matched controls, external baselines and paired significance tests.

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10–3.12, numpy 1.26+, scikit-learn 1.3+.

## 2. Quick start

```bash
python 00_selfcheck.py
```

The script checks that every data/model/result file is present and prints the global
boundary entropy of the four training sets. Expected output ends with `自检 通过` and
reports `{B: 3.0970, C: 2.5064, C': 2.7017, A: 2.2857}`.

## 3. Repository layout

```
.
├── 00_selfcheck.py                     artifact check + GBE of the four training sets
├── data/                               class maps, splits, filter decisions, embeddings
├── models/                             trained classifiers (5 MLP seeds + Ridge)
├── results/                            JSON results backing every table of the paper
│
├── Label-set construction
│   ├── 11_gen_round5.py                candidate generation, round 5 (294 long-tail classes)
│   ├── 12_build_round5_arms.py         arm assembly (plain / hybrid screening spaces)
│   ├── 13_eval_arms.py                 comparison of the assembled arms
│   ├── 14_gen_round6.py                candidate generation, round 6 (190 classes)
│   ├── 15_gen_round6b.py               candidate generation, round 6b (53 empty zero-shot classes)
│   ├── 16_build_round6_arms.py         round-6 assembly and full 679-class export
│   ├── 17_fill_underfilled.py          relaxed-staircase refill of under-filled classes
│   └── 18_gen_round7.py                round-7 targeted refill (dedup threshold 0.92)
│
├── Evaluation and model comparison
│   ├── 19_eval_final_matrix.py         four model groups on both test faces
│   ├── 20_eval_head_undersample.py     head-class undersampling study
│   ├── 21_eval_top100.py               per-class cap of 100 samples
│   ├── 22_eval_tail_final.py           long-tail behaviour of the final label set
│   ├── 24_rq5_rerun.py                 external baselines, full-data training
│   ├── 25_rq5_equal5.py                external baselines, matched per-class budget
│   ├── 26_multifamily.py               family-level topology under matched budgets
│   ├── 27_six_metrics.py               six topological metrics of the T1546 family
│   └── 30_common_matched.py            matched-sample-size comparison on 2,984 sentences
│
├── Diagnostics and sensitivity
│   ├── 04_audit_fidelity.py            definition fidelity of retained / removed / naive samples
│   ├── 06_eval_saturation.py           sample-size–accuracy saturation curve
│   ├── 23_anchor_sweep.py              intent-anchor injection strength sweep
│   ├── 29_zero_align.py                zero-shot alignment of class centroids
│   ├── 33_rebuild_texts.py             training-set statistics of all compared datasets
│   ├── 34_entropy_contrast.py          entropy in plain vs. intent-anchored space
│   ├── 35_sensitivity.py               entropy hyperparameter sensitivity
│   └── 37_gbe_error_analysis.py        GBE versus downstream error
│
├── Adaptive threshold and label-set audit
│   ├── 39_adaptive_threshold_pipeline.py   adaptive acceptance threshold (diagnosis–refill)
│   ├── 40_gbe_four_models.py               GBE of the four training sets
│   └── 41_tail_budget.py                   322 long-tail classes under matched budgets
│
└── Ablations
    ├── 42_ablation_no_structured.py    remove structured semantic decomposition
    ├── 43_ablation_entropy_fair.py     entropy selection vs. no selection, otherwise identical
    ├── 44_ablation_local_space.py      entropy selection in the local space vs. generation order
    ├── 47_v4_ablation.py               entropy vs. order vs. random selection at matched budget
    └── 48_entropy_smallbudget.py       entropy selection under tight budgets (5 and 10 per class)
```

Every script is runnable from the repository root and writes its output into `results/`.

## 4. Reproduction map

| Element of the paper | Script | Result file |
|---|---|---|
| Training-set statistics | `33_rebuild_texts.py` | `results.json` |
| Four model groups and GBE | `40_gbe_four_models.py`, `19_eval_final_matrix.py` | `gbe_four_models.json`, `final_matrix.json` |
| Matched-sample-size comparison (2,984 sentences) | `30_common_matched.py` | `common_matched.json` |
| Long-tail classes under matched budgets (322 classes) | `41_tail_budget.py` | `tail_budget_322.json` |
| Sample-size–accuracy saturation | `06_eval_saturation.py` | `saturation.json` |
| Component ablation (four components) | `42_ablation_no_structured.py`, `43_ablation_entropy_fair.py`, `44_ablation_local_space.py` | `ablation_no_structured.json`, `ablation_v4_fair.json` |
| Entropy selection vs. random at matched / tight budgets | `47_v4_ablation.py`, `48_entropy_smallbudget.py` | `ablation_v4_fair.json`, `entropy_smallbudget.json` |
| Entropy hyperparameter sensitivity | `35_sensitivity.py` | `sensitivity.json` |
| External baselines, full data and matched budget | `24_rq5_rerun.py`, `25_rq5_equal5.py` | `rq5_rerun.json`, `rq5_equalbudget5.json` |
| Six topological metrics of the T1546 family | `27_six_metrics.py` | `six_metrics.json` |
| Family-level topology under matched budgets | `26_multifamily.py` | `multifamily_matched.json` |
| Adaptive diagnosis–refill loop | `39_adaptive_threshold_pipeline.py` | `adaptive_threshold_pipeline.json` |
| Definition fidelity and zero-shot alignment | `04_audit_fidelity.py`, `29_zero_align.py` | `zero_align.json`, `ttests.json` |
| Entropy contrast across representation spaces | `34_entropy_contrast.py` | `entropy_contrast.json` |
| GBE versus downstream error | `37_gbe_error_analysis.py` | `gbe_error.json` |
| Head-class undersampling and the 100-sample cap | `20_eval_head_undersample.py`, `21_eval_top100.py` | `top100_matrix.json` |

Classifier training uses the fixed seeds 42–46; ablations and sensitivity analyses use 42–44.

## 5. Data artifacts

`data/` contains the class index, the feasible subset, the zero-shot list, the structured
attack graphs, the real-sentence test set, the per-candidate screening decisions and the
official-intent embeddings. `models/` contains the trained classifiers.

The large intermediate arrays are attached to the **v1.0-artifacts** release of this
repository:

| File | Content |
|---|---|
| `D_base_train_new.npz` | baseline set (11,840 × 768) |
| `D_naive_full.npz` | naive-LLM set (34,153) |
| `D_arms_HYBRID.npz` | oversampled pool screened in the hybrid space |
| `D_final.npz` | BEDR label set (24,174) |
| `orig4291_plain.npz` | full-coverage test set (4,291 rows, 679 classes) |
| `test2_full_plain.npz` | real-sentence test set built from the official procedure examples |
| `audit/embeds.npz`, `audit/texts.json` | embeddings and texts of the expert-audit sample |

Download them into `data/` (the two audit files into `data/audit/`), keeping the file
names, and every script becomes runnable.

## 6. License and citation

Code is released under the MIT license. The ATT&CK knowledge base used to build the
label set is available at <https://attack.mitre.org>. If you use this code or the
constructed label set, please cite the paper:

```bibtex
@article{yu2026bedr,
  title  = {BEDR: Boundary Entropy-Driven Resampling with Large Language Models for
            Fine-Grained Cyber Threat Intelligence Mapping},
  author = {Yu, Fengrui and Du, Yanhui},
  year   = {2026}
}
```
