**CIFAR-10 Image Classification with DenseNet121, ResNet20, and ResNet34**

This document describes the experimental setup, analysis scripts, and launchers for comparing optimizers on CIFAR-10 image classification. The experiments cover three architectures (DenseNet121, ResNet20, ResNet34) and evaluate both a full multi-optimizer comparison and a focused ablation of the EMOJI centering mechanism.

All scripts reside under the repository root `emoji_optimizer/classification_cifar10_resnet20`. Analysis scripts live at the relative paths listed below; training is launched via the PowerShell scripts in `schedulers/`.

### Statistical Convention

For every seed the representative value is the mean of the final 30 common epochs.

When multiple seeds are available, the reported mean and sample standard deviation (ddof=1) are computed across these seed-level final-30 means. Raw per-epoch values are never averaged across seeds.

All epoch-wise plots and CSVs are restricted to the intersection of epochs that exist for every seed of every optimizer under consideration. The final-30 summary always uses the last 30 epochs of this common intersection.

### EMOJI Naming Convention

* Emoji (Proposed) / α=1 (Emoji) = proposed configuration with centering enabled.  
* α=0 (No centering) = ablation only (centering disabled).  

α is used exclusively for the ablation study and is not a tunable hyperparameter of the proposed optimizer.

### Confirmed Architecture Settings

| Setting                   | ResNet-20          | ResNet-34          | DenseNet-121 variant |
| ------------------------- | ------------------ | ------------------ | -------------------- |
| Input channels            | 3                  | 3                  | 3                    |
| Input resolution          | 32 × 32 expected   | 32 × 32 expected   | 32 × 32 expected     |
| Stem convolution          | 3 × 3, 16 channels | 3 × 3, 64 channels | 3 × 3, 24 channels   |
| Block type                | BasicBlock         | BasicBlock         | Bottleneck           |
| Stage/block configuration | [3, 3, 3]          | [3, 4, 6, 3]       | [6, 12, 24, 16]      |
| Growth rate               | N/A                | N/A                | 12                   |
| Number of classes         | 10                 | 10                 | 10                   |
| Normalization             | BatchNorm2d        | BatchNorm2d        | BatchNorm2d          |
| Activation                | ReLU               | ReLU               | ReLU                 |
| Batch size                | 128                | 128                | 128                  |

### Experimental Protocol — ResNet-20

To evaluate the proposed EMOJI optimizer against AdamW, AdaBelief, Yogi, AdaBound, Lion, and SGDW, we conducted experiments on CIFAR-10 using the ResNet-20 architecture. All optimizers were evaluated under the same hardware and software conditions and with the same training configuration. The model was trained for 180 epochs with a batch size of 128, and each experiment was repeated using four independent random seeds. The mean performance across the repeated runs was used as the primary evaluation criterion.

For all optimizers, a common search range was considered for the learning rate, and the selected values were determined based on the validation performance obtained during the hyperparameter search. The parameter β₂ was similarly evaluated over the specified search range for optimizers with a corresponding parameter. The search ranges and selected values are summarized in Table S1.

Weight decay was implemented in a decoupled form for all evaluated optimizers. A value of 5×10⁻⁴ was used for AdamW, AdaBelief, Yogi, AdaBound, EMOJI, and SGDW, whereas the weight decay value for Lion was set to 5×10⁻³, following the configuration adopted from the corresponding reference.

To improve training stability, a linear warmup schedule was applied at the beginning of training for all optimizers. During the warmup phase, the learning rate was increased linearly from a very small initial value to the selected learning rate over 250 optimization steps. After warmup, a step-wise learning-rate scheduler was applied at epoch 110, reducing the learning rate by a factor of 10 for all evaluated methods.

For the numerical-stability constant, ε = 10⁻⁸ was used for AdamW, AdaBelief, AdaBound, and EMOJI. For Yogi, ε = 10⁻³ was used according to the reference configuration, and its initial second-moment accumulator was set to 10⁻⁶. For AdaBound, the optimizer-specific final_lr and γ parameters were set to 0.1 and 10⁻³, respectively. The complete hyperparameter search ranges and selected configurations are summarized in Table S1.

**Table S1. Hyperparameter search ranges and selected configurations of the evaluated optimizers on ResNet-20 with CIFAR-10.**

| Optimizer        | LR Search Range            | Selected LR | β₁  | β₂ Search Range | Selected β₂ | WD Type   | WD Value     | ε       |
| ---------------- | -------------------------- | ----------- | --- | --------------- | ----------- | --------- | ------------ | ------- |
| AdamW            | {0.1, 0.01, 0.001, 0.0001} | 0.01        | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻⁴       | 10⁻⁸    |
| AdaBelief        | {0.1, 0.01, 0.001, 0.0001} | 0.01        | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻⁴       | 10⁻⁸    |
| Yogi             | {0.1, 0.01, 0.001, 0.0001} | 0.01        | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻⁴       | 10⁻³    |
| EMOJI (Proposed) | {0.1, 0.01, 0.001, 0.0001} | 0.01        | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻⁴       | 10⁻⁸    |
| AdaBound         | {0.1, 0.01, 0.001, 0.0001} | 0.001       | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻⁴       | 10⁻⁸    |
| Lion             | {0.1, 0.01, 0.001, 0.0001} | 0.001       | 0.9 | {0.99, 0.999}   | 0.99        | Decoupled | 5×10⁻³       | —       |
| SGDW             | {0.1, 0.01, 0.001, 0.0001} | 0.1         | 0.9 | —               | —           | Decoupled | 5×10⁻⁴       | —       |

¹ Abbreviations: LR denotes learning rate, and WD denotes weight decay. For SGDW, the coefficient reported under β₁ is the momentum coefficient. For Lion, β₂ is an optimizer-specific smoothing coefficient rather than a second-moment parameter.

### Experimental Protocol — ResNet-34 and DenseNet-121

To further evaluate the applicability of the proposed EMOJI optimizer across different network architectures, we conducted experiments on CIFAR-10 using ResNet-34 and DenseNet-121. ResNet-34 was trained for 200 epochs, while DenseNet-121 was trained for 180 epochs. All experiments were performed with a batch size of 128 and repeated using four independent random seeds. The mean performance across the repeated runs was used as the primary evaluation criterion.

To maintain a consistent comparison while accounting for the different network architectures, the same optimizer configurations were used for both models. Rather than performing an extensive hyperparameter search for each optimizer and architecture, we adopted configurations based on established settings reported in the literature and the corresponding reference implementations. The optimizer configurations are summarized in Table S2.

For all optimizers, a linear warmup schedule was applied during the first 750 optimization steps. The learning rate was increased linearly during warmup until reaching the corresponding value specified in Table S2. After warmup, training continued using the selected learning rate. At epoch 140, a step-wise learning-rate scheduler reduced the learning rate by a factor of 10 for all optimizers, after which training continued with the reduced learning rates until the end of the respective training schedules.

Weight decay was applied in decoupled form. A value of 0.01 was used for AdamW, AdaBelief, Yogi, EMOJI, AdaBound, and SGDW, while Lion used a value of 0.1. For SGDW, the β₁ entry in Table S2 denotes the momentum coefficient rather than an exponential first-moment coefficient.

For Yogi, the initial second-moment accumulator was set to 10⁻⁶, following the configuration used in the corresponding reference. For AdaBound, the optimizer-specific final_lr and γ parameters were set to 0.1 and 10⁻³, respectively. The numerical-stability constants followed the protocol described in Section S2.1, with Yogi using ε = 10⁻³. The effect of varying ε, including the sensitivity of AdaBelief and EMOJI to alternative values, is investigated separately in Section S5.

**Table S2. Optimizer hyperparameter configurations used for ResNet-34 and DenseNet-121 on CIFAR-10.**

| Optimizer        | LR     | β₁  | β₂   | WD Type   | WD Value | ε     |
| ---------------- | ------ | --- | ---- | --------- | -------- | ----- |
| AdamW            | 0.001  | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸  |
| AdaBelief        | 0.001  | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸  |
| Yogi             | 0.001  | 0.9 | 0.999| Decoupled | 0.01     | 10⁻³  |
| EMOJI (Proposed) | 0.001  | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸  |
| AdaBound         | 0.001  | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸  |
| Lion             | 0.0001 | 0.9 | 0.99 | Decoupled | 0.1      | —     |
| SGDW             | 0.1    | 0.9 | —    | Decoupled | 0.01     | —     |

### Analysis Scripts

#### 1. Full multi-optimizer comparison

| Script                              | Model       |
| ----------------------------------- | ----------- |
| final_densenet121alloptimsfig.py.py | DenseNet121 |
| final_resnet20alloptimsfig.py.py    | ResNet20    |
| final_resnet34alloptimsfig.py       | ResNet34    |

Optimizers evaluated (4 seeds each):

* AdaBelief  
* Emoji (Proposed)  
* AdaBound  
* AdamW  
* Yogi  
* Lion  
* SGDW  

Key behaviour:

1. Compute the intersection of epochs present in all seeds of all optimizers.  
2. Restrict every curve and CSV to this common epoch range.  
3. Plot mean ± 1 sample STD across seeds.  
4. Compute the final-30-common-epoch summary (per-seed mean of the last 30 common epochs → mean ± sample STD across seeds).  

Outputs (written to `csv_summaries/` and current directory):

| File pattern                                | Description                                         |
| ------------------------------------------- | --------------------------------------------------- |
| {Model}_{tag}_STD.svg                       | Epoch-wise mean ± STD curves (common range only)    |
| {Model}_{tag}_common_summary.csv            | Epoch-wise mean, std, std_lower, std_upper, n_seeds |
| {Model}_final_30_common_epochs_mean_std.csv | Final summary: mean ± std across seeds              |

CSV columns for the final summary:
```
model, task, dataset, metric, optimizer, n_seeds, mean ± std
```

CSV columns for the epoch-wise summary:
```
epoch, optimizer, mean, std, std_lower, std_upper, n_seeds
```

#### 2. Ablation (centering on/off)

| Script                          | Model       |
| ------------------------------- | ----------- |
| ablation_meanstd95densnet121.py | DenseNet121 |
| ablation_meanstd95resnet20.py   | ResNet20    |
| ablation_meanstd95resnet34.py   | ResNet34    |

Configurations compared (4 seeds each):

* Original AdaBelief  
* α=0 (No centering)  
* α=1 (Emoji)  

Key behaviour:

* Same common-epoch intersection logic (when implemented) or direct final-30-epoch averaging.  
* Plots and CSVs follow the identical statistical convention (final 30 epochs → seed-level means → cross-seed mean ± sample STD).  

Outputs:

| File pattern                                                           | Description            |
| ---------------------------------------------------------------------- | ---------------------- |
| {model}_{tag}_STD.png                                                  | Epoch-wise mean ± STD  |
| {Model}_{tag}_summary.csv                                              | Epoch-wise mean/std    |
| {Model}_final_30_epochs_mean_std.csv or ..._common_epochs_mean_std.csv | Final 30-epoch summary |

### Training Launchers

Located in `schedulers/`.

| Launcher            | Model       |
| ------------------- | ----------- |
| run_densenet121.ps1 | DenseNet121 |
| run_resnet20.ps1    | ResNet20    |
| run_resnet34.ps1    | ResNet34    |

Each launcher runs the following optimizers for four seeds:

* emojiabl (α = 0)  
* emoji (α = 1, proposed)  
* adabelief  
* adamw  
* lion  
* sgdw  
* adaboundw  
* yogi  

The hyperparameter settings used by the launchers are fully consistent with the configurations reported in Tables S1 and S2 above.

### Output Directory Structure

```
csv_summaries/
├── DenseNet121_Accuracy_test_common_summary.csv
├── DenseNet121_final_30_common_epochs_mean_std.csv
├── ResNet20_...
├── ResNet34_...
└── ...

(and corresponding .svg / .png plots in the working directory)
```

All final numeric results that appear in tables or figures follow the final-30-common-epoch → seed-level mean → cross-seed mean ± sample STD protocol described above.

---

**Note on TensorBoard logs**

All TensorBoard event logs are stored inside the `runs/` directory.  
The six analysis scripts (`final_*` and `ablation_*`) read these TensorBoard logs, extract the scalar metrics, and compute the means and standard deviations.

To compare different optimizers on a **single seed**, simply copy the folder name that appears after `runs/` (without quotation marks) and paste it into TensorBoard. You can then overlay the curves of different optimizers for that specific seed.

---

**CIFAR-10 Dataset Structure**

For the CIFAR-10 experiments the data directory must have the following structure:

```
data/
├── cifar-10-python.tar.gz          # (optional – original archive)
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── readme.html
```

### Download Link

Official CIFAR-10 Python version:

**https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz**

### How to prepare

1. Download the archive from the link above.
2. Extract it. You will obtain the folder `cifar-10-batches-py`.
3. Place the extracted folder inside your project’s `data/` directory so that the structure matches the tree shown above.

