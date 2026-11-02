**ALBERT-base on GLUE (CoLA, MRPC, STS-B)**

This document describes the experimental setup, analysis scripts, and launchers for evaluating the proposed EMOJI optimizer against AdamW, AdaBelief, and Lion on three GLUE tasks using ALBERT-base.

All analysis scripts and the training launcher reside under the corresponding experiment directory. TensorBoard event logs are stored in the `logs/` directory. The analysis scripts read these logs to compute means and standard deviations.

---

### Statistical Convention

For every seed the representative value is the **mean of the final N common steps**, where N is task-specific and corresponds to the late stage of training:

- CoLA  → last **11** common steps  
- MRPC  → last **8** common steps  
- STS-B → last **7** common steps  

These final-N windows cover the concluding portion of each training run (approximately the last one-fifth of the logged evaluation points). 

When multiple seeds are available, the reported mean and sample standard deviation (ddof=1) are computed **across these seed-level final-N means**. Raw per-step values are never averaged across seeds.

All step-wise plots and CSVs are restricted to the intersection of steps that exist for every seed of every active optimizer. The final-N summary always uses the last N steps of this common intersection.

---

### EMOJI Naming Convention

- **Emoji (Proposed)** = proposed configuration with centering enabled.  
- α is used only for ablation studies and does not appear in the present GLUE experiments.

---
### Model Configuration (ALBERT-base-v2)

The following table summarizes the architecture and key settings of the ALBERT-base-v2 model used in all GLUE experiments (CoLA, MRPC, and STS-B). The model follows the standard ALBERT-base configuration with parameter sharing across layers and a factorized embedding parameterization.

| Setting                        | Value                          |
|--------------------------------|--------------------------------|
| Architecture                   | ALBERT-base-v2                 |
| Hidden size                    | 768                            |
| Number of attention heads      | 12                             |
| Number of hidden layers        | 12                             |
| Intermediate size              | 3072                           |
| Embedding size                 | 128                            |
| Vocabulary size                | 30 000                         |
| Maximum position embeddings    | 512                            |
| Classifier dropout             | 0.1                            |
| Attention / hidden dropout     | 0.0                            |

These settings were kept identical across all optimizers and all three tasks to ensure a controlled comparison.
### Experimental Protocol

To evaluate the proposed EMOJI optimizer across different aspects of natural language understanding, we conducted experiments on three GLUE tasks: CoLA, MRPC, and STS-B. We used the ALBERT-base-v2 model (Lan et al., 2020) as implemented in the Hugging Face Transformers library. These tasks cover complementary forms of language understanding, including grammatical acceptability, paraphrase detection, and semantic textual similarity. CoLA is formulated as a single-sentence binary classification task that evaluates grammatical acceptability, whereas MRPC is a sentence-pair classification task that determines whether two sentences are semantically equivalent. STS-B evaluates semantic similarity between sentence pairs as a continuous regression task, with performance measured using Pearson and Spearman correlation coefficients. Together, these tasks provide a compact evaluation across distinct aspects of language understanding while remaining computationally manageable under limited hardware resources.

The ALBERT-base model was evaluated using four optimizers: AdamW, Lion, AdaBelief, and the proposed EMOJI optimizer. Each optimizer was evaluated using **five independent random seeds** (1402, 1993, 2025, 8, 11021979). The same set of seeds was used across all four optimizers to ensure a consistent basis for comparison and to reduce the influence of stochastic variation. The mean performance across the five runs was used as the primary evaluation criterion.

The general ALBERT-base fine-tuning configuration was adopted from the original ALBERT configuration, including task-specific learning rates, dropout rates, training steps, warmup steps, and maximum sequence length. Because of GPU memory constraints, the batch size was set to **8 for all three tasks** (CoLA, MRPC, and STS-B). The remaining task-level training settings were kept unchanged.

For all four optimizers the ALBERT dropout rate was set to 0.0, the classifier dropout rate to 0.1, and the maximum sequence length to 512. The number of training steps and warmup steps was kept consistent with the corresponding ALBERT configuration for each task.

Optimizer-specific hyperparameters were configured according to the corresponding reference settings:

- AdamW and AdaBelief used β₁ = 0.9, β₂ = 0.999, and ε = 10⁻⁸.  
- EMOJI used the same β₁, β₂, and ε values.  
- Lion used β₁ = 0.9 and β₂ = 0.99.

Lion was configured according to the recommendations established in the original Lion literature and followed consistently in all previous experiments. Because Lion uses a sign-based update, the magnitude of its parameter updates tends to be larger than that of Adam-style optimizers. Consequently:

- The learning rate for Lion was set to **one order of magnitude smaller** than the corresponding Adam-family learning rate (e.g., 10⁻⁶ vs 10⁻⁵ on CoLA, 2×10⁻⁶ vs 2×10⁻⁵ on MRPC and STS-B).  
- The weight-decay coefficient was increased (0.1 instead of 0.01) to compensate for the larger update norms and to maintain training stability.

These relative scalings of learning rate and weight decay are optimizer-specific recommendations rather than a requirement that every optimizer share identical hyper-parameters. All other training settings (architecture, dropout, number of steps, warmup, maximum sequence length, etc.) remained identical across the four optimizers.

---

---

### Hyperparameter Configurations

**Table S1. Hyperparameter configurations used for AdamW, Lion, AdaBelief, and EMOJI on the CoLA task using ALBERT-base.**

| Optimizer        | LR     | BSZ | ALBERT DR | Classifier DR | TS   | WS  | MSL | WD   | β₁  | β₂   | ε     |
|------------------|--------|-----|-----------|---------------|------|-----|-----|------|-----|------|-------|
| AdamW            | 10⁻⁵  | 8   | 0.0       | 0.1           | 5336 | 320 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| Lion             | 10⁻⁶  | 8   | 0.0       | 0.1           | 5336 | 320 | 512 | 0.1  | 0.9 | 0.99 | —     |
| AdaBelief        | 10⁻⁵  | 8   | 0.0       | 0.1           | 5336 | 320 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| EMOJI (Proposed) | 10⁻⁵  | 8   | 0.0       | 0.1           | 5336 | 320 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |

**Table S2. Hyperparameter configurations used for AdamW, Lion, AdaBelief, and EMOJI on the MRPC task using ALBERT-base.**

| Optimizer        | LR       | BSZ | ALBERT DR | Classifier DR | TS  | WS  | MSL | WD   | β₁  | β₂   | ε     |
|------------------|----------|-----|-----------|---------------|-----|-----|-----|------|-----|------|-------|
| AdamW            | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 800 | 200 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| Lion             | 2×10⁻⁶  | 8   | 0.0       | 0.1           | 800 | 200 | 512 | 0.1  | 0.9 | 0.99 | —     |
| AdaBelief        | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 800 | 200 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| EMOJI (Proposed) | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 800 | 200 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |

**Table S3. Hyperparameter configurations used for AdamW, Lion, AdaBelief, and EMOJI on the STS-B task using ALBERT-base.**

| Optimizer        | LR       | BSZ | ALBERT DR | Classifier DR | TS   | WS  | MSL | WD   | β₁  | β₂   | ε     |
|------------------|----------|-----|-----------|---------------|------|-----|-----|------|-----|------|-------|
| AdamW            | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 3598 | 214 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| Lion             | 2×10⁻⁶  | 8   | 0.0       | 0.1           | 3598 | 214 | 512 | 0.1  | 0.9 | 0.99 | —     |
| AdaBelief        | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 3598 | 214 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |
| EMOJI (Proposed) | 2×10⁻⁵  | 8   | 0.0       | 0.1           | 3598 | 214 | 512 | 0.01 | 0.9 | 0.999| 10⁻⁸ |

**Abbreviations:** LR, learning rate; BSZ, batch size; ALBERT DR, ALBERT dropout rate; Classifier DR, classifier dropout rate; TS, training steps; WS, warmup steps; MSL, maximum sequence length; WD, weight decay.


### Analysis Scripts

| Script                   | Task  | LAST_N_STEPS | Metrics                                              |
|--------------------------|-------|--------------|------------------------------------------------------|
| `final_cola_alloptim.py` | CoLA  | 11           | train/loss, eval/loss, eval/matthews_correlation     |
| `final_mrpc_alloptim.py` | MRPC  | 8            | train/loss, eval/loss, eval/accuracy, eval/f1        |
| `final_sts_alloptim.py`  | STS-B | 7            | train/loss, eval/loss, eval/pearson, eval/spearmanr  |

**Active optimizers in the scripts** (paths currently populated):

- AdaBelief  
- Emoji (Proposed)  
- AdamW  
- Lion  

(AdaBound, Yogi, and SGDW entries are present but empty.)

**Key behaviour of each script:**

1. Compute the intersection of steps present in all seeds of all active optimizers.  
2. Restrict every curve and CSV to this common step range.  
3. Plot mean ± 1 sample STD across the five seeds.  
4. Compute the final-N-common-step summary (per-seed mean of the last N common steps → mean ± sample STD across seeds).  

**Outputs (written to `csv_summaries/` and current directory):**

| File pattern                                         | Description                                      |
|------------------------------------------------------|--------------------------------------------------|
| `ALBERT_{task}_{tag}_STD.svg`                        | Step-wise mean ± STD curves                      |
| `ALBERT_{task}_{tag}_summary.csv`                    | Step-wise mean, std, std_lower, std_upper, n_seeds |
| `ALBERT_{task}_final_{N}_common_epochs_mean_std.csv` | Final summary: mean ± std across seeds           |

---

### Training Launcher

`run_ALBERT_FN.ps1`

Launches the four optimizers (AdaBelief, AdamW, Emoji, Lion) for each of the three tasks using the five seeds listed above. The command-line arguments are fully consistent with the hyperparameters reported in Tables S1–S3.

---

### Note on TensorBoard Logs

All TensorBoard event logs are stored inside the `logs/` directory.  
The three analysis scripts read these logs, extract the scalar metrics, and compute the means and standard deviations.

To compare different optimizers on a **single seed**, simply copy the folder name that appears after `logs/` (without quotation marks) and paste it into TensorBoard. You can then overlay the curves of different optimizers for that specific seed.

---

### Model Configuration (ALBERT-base-v2)

- Architecture: AlbertForSequenceClassification  
- Hidden size: 768  
- Number of attention heads: 12  
- Number of hidden layers: 12  
- Intermediate size: 3072  
- Embedding size: 128  
- Vocabulary size: 30 000  
- Maximum position embeddings: 512  
- Classifier dropout: 0.1  
- Attention / hidden dropout: 0.0


