---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-2B-Base
base_model_relation: finetune
tags:
- decision-model
- classification
- calibration
- qwen3.5
- jev-style
- single-prefill
---
# Jev-Style-Qwen3.5-2B-Decision v2 (HF BF16)

**Website:** [jevstyle.com](https://jevstyle.com/#v2) — all JevStyle decision models, benchmarks and quickstart in one place.

A **Jev-style decision model** for classification, routing and typed choices. Give it a state, a question and a list of options; one prefill returns a selected option **with calibrated probabilities**.

| Build | Weight size | Inference |
|---|---:|---|
| **HF BF16 · this repository** | 3.76 GB | Transformers + decision client |
| [GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF) | 1.27–3.78 GB | Q4_K_M / Q8_0 / BF16 · llama.cpp |
| [MLX BF16](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-MLX-bf16) | 3.76 GB | Apple Silicon + native MLX client |

**Download this build:** [model.safetensors](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2/resolve/main/model.safetensors?download=true). The repository also includes its calibration, inference client and evaluation records.

## Results

**81.20% macro accuracy on the fixed English reference panel**, compared with 76.68% for v1 and 75.09% for English Laya. The results below use the CUDA reference structure: 11 real-label task groups, 3,277 decisions, equal task weights, and the same 3,100-record calibration split. Results for the released deployment formats appear further below.

| Metric | Jev-Style v1 | English Laya | **Jev-Style v2** |
|---|---:|---:|---:|
| Accuracy ↑ | 76.68% | 75.09% | **81.20%** |
| Macro-F1 ↑ | 75.42% | 73.45% | **79.78%** |
| Negative log-likelihood ↓ | 0.5752 | 0.6318 | **0.5154** |
| Brier score ↓ | 0.3290 | 0.3482 | **0.2787** |

![Accuracy, Macro-F1, NLL and Brier comparison on the frozen English panel](figures/benchmark.png)

- **Higher accuracy:** +4.53 percentage points over v1 and +6.12 over English Laya; paired 95% intervals are [+3.58, +5.52] and [+4.64, +7.52] points, respectively, within this fixed panel.
- **Broader task coverage:** accuracy point estimates ahead of English Laya in **9 of 12 task groups**, including the separately scored teacher-reference typed-decisions group.
- **Better probability quality against English Laya:** **18.4% lower NLL**, **20.0% lower Brier score**, and **26.4% lower task-macro ECE**.
- **Efficient adaptation:** **36.9 minutes of main training on one H100 80GB**, using rank-32 LoRA on a 2B-class text backbone.

## Calibration

The reliability diagram plots the v2 model's stated confidence against observed correctness. Every real-label evaluation decision is included; the histogram shows how many predictions fall in each confidence bin. Error bars show Wilson 95% intervals. The accompanying ECE comparison averages per-task calibration errors.

![Reliability diagram with confidence counts and task-macro ECE comparison](figures/calibration.png)

Temperature is fitted on the calibration split. HF and MLX clients apply the supplied calibration automatically; the calibrated GGUF file incorporates it in the final normalization tensor.

## Robustness

**Option-order flip rate is halved relative to English Laya**, with **80.00% accuracy after permutation** on the same 400 Choice/Bool decisions. Semantic options are mapped back to their original identities before scoring.

| Option-permutation test | Jev-Style v1 | English Laya | **Jev-Style v2** |
|---|---:|---:|---:|
| Decision flip rate ↓ | 9.25% | 12.00% | **6.00%** |
| Accuracy after permutation ↑ | 66.75% | 67.00% | **80.00%** |

![Option-order stability and accuracy after permutation](figures/robustness.png)

On a separate **200-pair programmatic threshold-policy test**, both decisions in a counterfactual pair are correct in **71.50%** of pairs for v2, compared with 63.00% for v1. This test measures that specific rule family.

## Task-level results

<details>
<summary><strong>Per-task accuracy: all 11 real-label tasks and the separate typed-decision group</strong></summary>

| Real-label task | Examples | Jev-Style v1 | English Laya | Jev-Style v2 |
|---|---:|---:|---:|---:|
| AG News | 300 | 87.67% | 89.00% | 88.00% |
| ANLI | 300 | 48.00% | 49.67% | 48.67% |
| BoolQ | 300 | 82.67% | 75.67% | 81.67% |
| Emotion | 300 | 58.33% | 60.33% | 85.33% |
| Enron spam | 300 | 77.33% | 96.33% | 97.67% |
| HANS | 300 | 68.00% | 75.00% | 68.00% |
| IMDb | 300 | 96.67% | 93.67% | 96.33% |
| MNLI | 300 | 86.67% | 85.00% | 88.00% |
| RTE | 277 | 84.48% | 77.98% | 85.92% |
| SST-2 | 300 | 92.67% | 91.67% | 93.00% |
| SST-5 | 300 | 61.00% | 31.67% | 60.67% |

The separate typed-decisions group contains 2,000 teacher-reference decisions from 400 states. Teacher agreement is 53.35% for v1, 37.55% for English Laya and **73.45% for v2** under the fixed primary interface. This group is excluded from the real-label macro. The comparison here uses the English Laya checkpoint; specialist-checkpoint and rendering sensitivity results are provided in [baseline_sensitivity.json](evaluation/baseline_sensitivity.json).

</details>


## Deployment validation

GGUF is available in **Q4_K_M, Q8_0 and BF16**, each with its own calibration and verification record.

| Released format | Weight size | Validated result | Evaluation set |
|---|---:|---|---|
| HF BF16 | 3.76 GB | **81.27%** real-label macro accuracy | Full 3,277 real-label decisions |
| Native MLX BF16 | 3.76 GB | **99.6%** choice agreement with CUDA BF16 | Frozen 500-decision deployment subset |
| Calibrated GGUF Q8_0 | 2.01 GB | **99.2%** choice agreement with CUDA BF16 | Same 500-decision deployment subset |
| Calibrated GGUF Q4_K_M | 1.27 GB | **91.4%** choice agreement with CUDA BF16 | Same 500-decision deployment subset |
| Calibrated GGUF BF16 | 3.78 GB | **99.6%** choice agreement with CUDA BF16 | Same 500-decision deployment subset |

Each deployment format has its own validation record. Native MLX packaging reproduces the verified MLX client's logits exactly on all 500 deployment cases. The Q8_0 model is approximately **46.7% smaller** than the BF16 GGUF export.

On the same 500-case deployment subset, real-label task-macro accuracy is **79.10%** for CUDA BF16, **79.04%** for MLX BF16 and **78.69%** for Q8_0. Full-panel reference results and deployment-subset results use their respective denominators.

<details>
<summary><strong>Evaluation data and downloadable vector charts</strong></summary>

- [Reference metrics and paired intervals](evaluation/reference_comparison.json)
- [Deployment validation](evaluation/deployment.json)
- [Baseline sensitivity results](evaluation/baseline_sensitivity.json)
- [Data sources and split manifest](evaluation/data_manifest.json)
- [Chart data, confidence bins and sample counts](evaluation/chart_data.json)
- Vector charts: [benchmark](figures/benchmark.svg), [calibration](figures/calibration.svg), [robustness](figures/robustness.svg)

The benchmark figures describe the fixed CUDA reference comparison. Reliability pools all real-label examples into confidence bins; task-macro ECE is the mean of 11 separate task ECE values. These are distinct aggregations. The 9/12 figure counts task-level point estimates. Individual prediction probabilities, task summaries, test protocols and calibration records were retained when drawing these charts.

</details>

## Quick start

```bash
python -m pip install -U huggingface_hub
hf download chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2 --local-dir jev-v2
cd jev-v2
python -m pip install -r requirements.txt
python -m jev_h100.decide --model . --device cpu \
  --state "The film was excellent." \
  --question "What is the sentiment of this review?" \
  --options negative positive
```

The CPU command is a portable smoke example. For H100 CUDA inference, the validated environment additionally used Triton 3.7.1, flash-linear-attention 0.5.2, fla-core 0.5.2 and causal-conv1d 1.7.0, with a wheel matching Torch/CUDA; use `--device cuda` in that environment. The included `jev_h100` loader preserves the trained Q/K normalization convention and projects only declared-option rows in FP32.

On Apple Silicon, use the dedicated [MLX BF16 repository](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-MLX-bf16) or install `requirements-mlx.txt` and run the included `jev_mlx_client.py --model .` with the same state, question and options. Its MLX calibration is selected automatically.


## Decision interface

Provide an English state, a question, and **2–26 unique options**, within a **1,024-token prompt**. A single prefill produces one logit per declared option. Apply the supplied calibration once and normalize over those options to obtain the decision probabilities.

```text
You are a decision function. Read the state, then answer the question by choosing exactly one option.

[State]
The film was excellent.

[Question]
What is the sentiment of this review?

[Options]
A. negative
B. positive

Answer:
```

The supplied clients implement this exact prompt and read the next-position ` A` through ` Z` token scores. Use this decision interface for Choice, Bool and ordered Score tasks; `decide_bool` returns the probability of yes, and `decide_score` also returns the expected zero-based level. For ordered scores, supply options from lowest to highest.

## Training

Continued from the uncalibrated Jev-Style v1 text backbone derived from [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). Training used a BF16 backbone, FP32 rank-32 LoRA (alpha 32), 186 adapted modules and 33,638,400 trainable parameters. Effective batch size was 64 with a 1,024-token budget.

The 60,000-record training pool combines original-task replay with emotion, email, typed workflow decisions, label transformations and programmatic threshold rules. A two-stage schedule increases hard-example sampling while retaining approximately 50% original-task replay. Main training completed 1,000 optimizer updates and processed 11,605,632 tokens in 36.9 minutes on one H100 80GB.

Development (2,050 records), calibration (3,100 records) and final evaluation (5,277 decisions) were handled separately. Checkpoint selection used development results; calibration used the calibration split. This release records one training seed. The evaluation JSON files document the dataset, rendering and deployment protocols.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE). This release builds on Qwen3.5-2B-Base and Jev-Style v1. Training data retain their original source licenses; source and split details are recorded in the data manifest. The release contains model artifacts and aggregate evaluation records.

## Contact

I welcome internship, employment, and research collaboration opportunities. Please contact me at [**yanchaoliang369@gmail.com**](mailto:yanchaoliang369@gmail.com).

欢迎提供实习、工作及科研合作机会，请邮件联系：[yanchaoliang369@gmail.com](mailto:yanchaoliang369@gmail.com)。
