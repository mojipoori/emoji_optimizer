**ImageNet-1K Classification with ResNet-18**

This document describes the experimental setup for evaluating the proposed EMOJI optimizer against AdaBelief on ImageNet-1K using the ResNet-18 architecture.

---

### Experimental Protocol

To further evaluate the scalability of the proposed EMOJI optimizer on a larger-scale image classification task, we conducted experiments on ImageNet-1K using the ResNet-18 architecture for 100 epochs with a batch size of 256. Due to the substantially higher computational cost of ImageNet-1K training, the evaluation was limited to the proposed EMOJI optimizer and AdaBelief. Each optimizer was evaluated in a **single training run** under the same experimental configuration and using the same random seed (seed = 42). Accordingly, the results in this section are presented as a direct comparison between the two runs rather than as estimates aggregated across multiple independent seeds.

Rather than performing an extensive hyperparameter search for this dataset, the optimizer configurations were selected based on standard values reported in the literature and, in particular, the AdaBelief reference configuration for the same architecture and dataset.

For both AdaBelief and EMOJI, the learning rate, β₁, β₂, and weight decay were configured according to the values summarized in Table S1. The numerical-stability constant was set to ε = 10⁻⁸ for both optimizers, consistent with the AdaBelief configuration reported for ResNet-18 on ImageNet-1K.

A step-wise learning-rate schedule was applied during training, with the learning rate reduced by a factor of 10 at epochs 30, 60, and 90. This schedule allows the optimizers to proceed with progressively smaller updates during the intermediate and later stages of training as optimization approaches lower-loss regions.

---
### Model Architecture (ResNet-18)

The experiments use the standard ResNet-18 architecture as implemented in `torchvision.models.resnet18`. The network consists of an initial 7×7 convolution followed by four residual stages with BasicBlocks and a final fully-connected classification layer.

| Setting                        | Value                          |
|--------------------------------|--------------------------------|
| Architecture                   | ResNet-18                      |
| Input channels                 | 3 (RGB)                        |
| Stem convolution               | 7×7, 64 channels               |
| Block type                     | BasicBlock                     |
| Stage configuration            | [2, 2, 2, 2]                   |
| Channel progression            | 64 → 128 → 256 → 512           |
| Number of classes              | 1000                           |
| Normalization                  | BatchNorm2d                    |
| Activation                     | ReLU                           |
| Approximate number of parameters | 11.7 million                 |

All residual stages use the standard BasicBlock design with two 3×3 convolutions. Downsampling is performed by 1×1 convolutions in the first block of stages 2, 3, and 4. The final feature map is globally average-pooled and passed through a linear layer that produces logits for the 1000 ImageNet classes.

---

### Hyperparameter Configurations

**Table S1. Optimizer hyperparameter configurations used for ResNet-18 on ImageNet-1K.**

| Optimizer        | LR    | β₁  | β₂   | WD Type   | WD Value | ε     |
|------------------|-------|-----|------|-----------|----------|-------|
| EMOJI (Proposed) | 0.001 | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸ |
| AdaBelief        | 0.001 | 0.9 | 0.999| Decoupled | 0.01     | 10⁻⁸ |

**Abbreviations:** LR = learning rate; WD = weight decay.

---

### Training Configuration

| Setting                  | Value                                      |
|--------------------------|--------------------------------------------|
| Architecture             | ResNet-18                                  |
| Dataset                  | ImageNet-1K                                |
| Number of epochs         | 100                                        |
| Batch size               | 256                                        |
| Learning-rate schedule   | Step decay (×0.1) at epochs 30, 60, and 90 |
| Random seed              | 42                                         |
| Weight decay type        | Decoupled                                 |
| Warmup                   | Disabled                                   |

---

### Dataset Folder Structure

The ImageNet-1K data must be organized into three distinct folders with specific structures, as required by the training and evaluation scripts.

#### 1. Training folder (`ILSVRC2012_img_train`)

This folder must follow the standard **ImageFolder** structure (one subdirectory per class):

```
ILSVRC2012_img_train/
├── n01440764/
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   └── ...
├── n01443537/
│   └── ...
├── n01484850/
│   └── ...
└── ...                    (1,000 class folders in total)
```

- Each class folder is named with its official ImageNet WNID (e.g., `n01440764`).
- Only images belonging to that class are placed inside the corresponding folder.

#### 2. Validation folder used during training (`ILSVRC2012_img_val_0`)

This folder is also organized as an **ImageFolder**:

```
ILSVRC2012_img_val_0/
├── n01440764/
│   ├── ILSVRC2012_val_00000293.JPEG
│   └── ...
├── n01443537/
│   └── ...
└── ...                    (1,000 class folders in total)
```

- Used to compute Top-1 and Top-5 validation accuracy after every epoch during the 100-epoch training run.
- Usually constructed from the official validation images with the help of `ILSVRC2012_validation_ground_truth.txt` and `meta.mat`.

#### 3. Flat validation folder for final evaluation (`ILSVRC2012_img_val`)

This folder must be **flat** (no class subdirectories):

```
ILSVRC2012_img_val/
├── ILSVRC2012_val_00000001.JPEG
├── ILSVRC2012_val_00000002.JPEG
├── ILSVRC2012_val_00000003.JPEG
├── ...
└── ILSVRC2012_val_00050000.JPEG
```

**Critical requirements:**
- Exactly 50,000 JPEG files.
- Filenames must follow the exact pattern `ILSVRC2012_val_00000001.JPEG` … `ILSVRC2012_val_00050000.JPEG`.
- No class subfolders are allowed.

#### Required accompanying files for final evaluation

These files must be present in the project root (or correctly referenced via command-line arguments):

```
.
├── Google_ReaL_labels.json
├── ILSVRC2012_validation_ground_truth.txt
└── ILSVRC2012_devkit_t12/
    └── data/
        ├── ILSVRC2012_validation_ground_truth.txt
        └── meta.mat
```

| File | Purpose |
|------|---------|
| `Google_ReaL_labels.json` | Google ReaL multi-annotator labels (50,000 entries) |
| `ILSVRC2012_validation_ground_truth.txt` | Official ImageNet ground-truth labels |
| `meta.mat` | Mapping from synset IDs to class indices (0–999) |

---

### Evaluation Protocol

Models were evaluated on the official ImageNet-1K validation set (50,000 images) using two complementary metrics:

1. **Original ImageNet accuracy** — Top-1 and Top-5 accuracy against the official ImageNet ground-truth labels.
2. **Google ReaL accuracy** — Top-1 and Top-5 accuracy against the multi-annotator ReaL labels (non-empty entries only).

Checkpoints were evaluated at epochs 30, 60, 90, the best validation checkpoint, and the final epoch (100).

---

### Analysis Script

`final_resnet18alloptimsfig.py`

- Reads the per-epoch CSV logs produced during training.
- Plots Top-1 and Top-5 validation accuracy curves.
- Computes the mean of the last 10 epochs for both Top-1 and Top-5 accuracy.
- Saves summary CSVs and SVG figures.

**Outputs:**

| File | Description |
|------|-------------|
| `ResNet18_Accuracy_validation (Top-1).svg` | Top-1 accuracy curves |
| `ResNet18_Accuracy_validation (Top-5).svg` | Top-5 accuracy curves |
| `csv_summaries/ResNet18_top1_top5_last10_mean.csv` | Final Top-1 / Top-5 summary (last 10 epochs) |
| `csv_summaries/ResNet18_*_summary.csv` | Epoch-wise mean summaries |

---

### Training and Evaluation Launchers

| File | Purpose |
|------|---------|
| `run.ps1` | Launches training for AdaBelief and EMOJI (seed 42) |
| `imagenet_evaluation.ps1` | Evaluates checkpoints (epochs 30/60/90, best, and final) using Original + ReaL metrics |

---

### Final Evaluation Results

After training, the checkpoints of both optimizers were re-evaluated on the local hardware (RTX 3070 Ti) using the official flat ImageNet validation set together with the Google ReaL labels. The evaluation was performed with the script `imagenet_evaluation.ps1`, which calls `main_when_copy_2.py` in evaluation mode (`--evaluate`) under the same configuration used during the original experiments.

The full checkpoint base names are:

- **EMOJI (Proposed):**  
  `model-resnet18-optimizer-emoji-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-emoji_eps1e-08-emoji_beta10.9-emoji_beta20.999-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False`

- **AdaBelief:**  
  `model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False`

The following checkpoints, saved during training, were evaluated:

**EMOJI (Proposed)**

| Checkpoint File          | Corresponding Epoch / Meaning            |
|--------------------------|------------------------------------------|
| `...-epoch-29`           | Epoch 30                                 |
| `...-epoch-59`           | Epoch 60                                 |
| `...-epoch-89`           | Epoch 90                                 |
| `..._model_best.pth.tar` | Best validation accuracy during training |
| `...(no suffix)`         | Final checkpoint (Epoch 100)             |

**AdaBelief**

| Checkpoint File          | Corresponding Epoch / Meaning            |
|--------------------------|------------------------------------------|
| `...-epoch-29`           | Epoch 30                                 |
| `...-epoch-59`           | Epoch 60                                 |
| `...-epoch-89`           | Epoch 90                                 |
| `..._model_best.pth.tar` | Best validation accuracy during training |
| `...(no suffix)`         | Final checkpoint (Epoch 100)             |

The complete evaluation outputs (Original ImageNet and Google ReaL Top-1 / Top-5 accuracies) for all of the above checkpoints are recorded in the following log files:

- `emoji_reallabel_evaluation.txt`
- `adabelief_reallabel_evaluation.txt`