# GPT-style Transformer Language Modeling on WikiText-2

We evaluated four optimization algorithms—AdamW, AdaBelief, EMOJI, and Lion—on autoregressive language modeling using WikiText-2. All experiments used the same GPT-style Transformer architecture, dataset splits, training configuration, and set of three shared random seeds to support a consistent comparison across optimizers.

### Script Locations (relative to repository root)

```text
├── fairseq/
│   ├── run_wiki_tr_gpt_loop.ps1
│   └── plot_wikitext2_gpt_metrics_per_seed.py
└── checkpoints/
    ├── plot_wiki2_gpt_abl_all.py
    ├── plot_wiki2_gpt_abl.py
    └── plot_wiki2_gpt.py
```

## Model architecture and training configuration

The model was implemented using Fairseq's `transformer_lm_gpt` architecture. It comprised four Transformer decoder layers, an embedding dimension of 128, four attention heads per layer, and a feed-forward network dimension of 512. The model used GELU activation, dropout of 0.1, attention dropout of 0.1, shared input and output embeddings, and a maximum sequence length of 128 tokens.

**Table 1. Model architecture and configuration.**

| Component | Configuration |
|---|---|
| Task | Autoregressive language modeling |
| Dataset | WikiText-2 |
| Fairseq architecture | `transformer_lm_gpt` |
| Transformer decoder layers | 4 |
| Embedding dimension | 128 |
| Attention heads per layer | 4 |
| Attention head dimension | 32 |
| Feed-forward dimension | 512 |
| Activation function | GELU |
| Input/output embeddings | Shared |
| Maximum sequence length | 128 tokens |
| Model dropout | 0.1 |
| Attention dropout | 0.1 |

The shared training configuration is summarized in Table 2.

**Table 2. Shared training configuration.**

| Training parameter | Value |
|---|---|
| Maximum epochs | 50 |
| Maximum tokens per batch | 1,024 |
| Gradient accumulation (`update_freq`) | 4 |
| Learning-rate scheduler | Inverse square root |
| Warm-up updates | 1,000 |
| Warm-up initial learning rate | `1e-7` |
| Gradient clipping (`clip_norm`) | 0.0 (disabled) |
| Logging interval | 100 updates |
| Data-loader workers | 0 |
| Random seeds | Three shared seeds |

The inverse-square-root schedule increased the learning rate during warm-up and subsequently decayed it according to the inverse square root of the update count. All optimizers used the same training schedule and model configuration, while their optimizer-specific hyperparameters were set as described below.

## Optimizer configurations

AdamW, AdaBelief, and EMOJI were evaluated using the same learning rate of `5e-4`, weight decay of `0.01`, and moment coefficients β₁ = 0.9 and β₂ = 0.98. The choice of these coefficients follows the original Transformer paper, *Attention Is All You Need*, which used Adam with β₁ = 0.9, β₂ = 0.98. In our experiments, we retained the same beta coefficients and used ε = 1e-8 and a common inverse-square-root learning-rate schedule with 1,000 warm-up updates. Applying the same settings to AdamW, AdaBelief, and EMOJI provides a controlled comparison of their optimization behavior under a shared configuration.

**Table 3. Optimizer-specific hyperparameters.**

| Optimizer | Learning rate | β₁ | β₂ | Epsilon | Weight decay |
|---|---|---|---|---|---|
| AdamW | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| AdaBelief | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| EMOJI | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| Lion | `5e-5` | 0.95 | 0.98 | Not applicable | 0.1 |

Lion was configured according to recommendations in the Lion literature. Its learning rate was set to one-tenth of the AdamW learning rate, while its weight decay was increased tenfold. This adjustment is motivated by Lion's sign-based update, which can produce updates with larger norms than AdamW's and therefore commonly calls for a smaller learning rate and a larger weight-decay coefficient. The coefficients β₁ = 0.95 and β₂ = 0.98 are also consistent with settings investigated for Lion in language-modeling experiments. These values are optimizer-specific recommendations rather than the default beta values of Lion.

The Lion configuration therefore follows the recommended relative scaling of learning rate and weight decay, rather than enforcing identical hyperparameters across all optimizers. Consequently, the comparison evaluates the four methods under the specified configurations, not under a strict requirement that every optimizer use identical optimizer-specific hyperparameters.

## Reproducibility

Each optimizer was evaluated using the same three random seeds, with all other shared model and training settings held constant. This shared-seed protocol helps control for variability due to random initialization and training stochasticity when comparing results across optimizers.

# WikiText-2 GPT-style Transformer: Optimizer Comparison and Ablation Study

The project contains four Python analysis scripts for analyzing optimizer performance in autoregressive language modeling using a GPT-style Transformer and the WikiText-2 dataset with Fairseq.

The scripts read existing Fairseq `train.log` files, extract epoch-wise training and validation metrics, calculate statistics across random seeds, export numerical results to CSV files, and generate plots in SVG and PNG formats. The per-seed plotting script additionally generates separate plots for each random seed and exports the same three timestamped CSV files.

These scripts are intended for post-training analysis. **They do not start training or modify checkpoints.**

All file paths below are relative to the project root unless otherwise specified.

## Scripts

### Comprehensive optimizer comparison

**Script:** `plot_wiki2_gpt_abl_all.py`

**Relative path:** `checkpoints\plot_wiki2_gpt_abl_all.py`

This script compares five optimizer configurations:

- **AdamW**
- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **Lion**
- **EMOJI (Proposed; α = 1)**
- **EMOJI (α = 0, Without Centering)**

The experiment compares the proposed EMOJI optimizer against three baseline optimizers—AdamW, AdaBelief, and Lion—and also includes the EMOJI variant without centering.

The script extracts the available training and validation metrics from the corresponding Fairseq logs and generates plots and CSV files for all five configurations across the specified random seeds.

### Sequential ablation study

**Script:** `plot_wiki2_gpt_abl.py`

**Relative path:** `checkpoints\plot_wiki2_gpt_abl.py`

This script focuses on three configurations:

- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **EMOJI (α = 0, Without Centering)**, which removes ε from the recursive second-moment state, and does not center the prediction error.
- **EMOJI (α = 1, Proposed)**, which incorporates two modifications: removing ε from the recursive second-moment state and centering the prediction error.

For clarity, α is not a hyperparameter of EMOJI and is introduced only in the ablation study to represent the presence or absence of the centering operation. The second-moment state in the ablation formulation is expressed as

$$
j_t = \mathrm{EMA}\big((d_t - \alpha v_t)^2\big).
$$

The EMOJI (α = 1, Proposed) configuration uses the centered gradient prediction error, whereas the No Centering variant omits this centering operation. Setting α = 0 therefore removes the centering term, making the resulting update theoretically equivalent to AdaBelief without ε in the recursive second-moment state. The proposed EMOJI configuration corresponds to the centered formulation.

### Four-optimizer comparison

**Script:** `plot_wiki2_gpt.py`

**Relative path:** `checkpoints\plot_wiki2_gpt.py`

This script compares four optimizer configurations:

- **AdamW**
- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **Lion**
- **EMOJI (Proposed; α = 1)**

Unlike the comprehensive comparison script, this script excludes the EMOJI no-centering variant from the active optimizer list. Its configuration remains commented out in the code.

The script processes the recorded training and validation metrics for the four active configurations, calculates statistics across the available seeds, and exports the same three types of CSV files and up to four SVG and four PNG figures.

### Per-seed plotting and CSV exports

**Script:** `plot_wikitext2_gpt_metrics_per_seed.py`

**Relative path:** `fairseq/plot_wikitext2_gpt_metrics_per_seed.py`

This script analyses the existing Fairseq `train.log` files for all five optimizer configurations and the three random seeds (`1993`, `1402`, and `2025`). It does not start training or modify checkpoints.

Unlike the aggregate plotting scripts, this script generates separate plots for each seed. Each plot compares the five optimizer configurations using the same seed, without averaging across seeds or displaying cross-seed standard-deviation bands:

- AdamW
- AdaBelief
- Lion
- EMOJI (α = 0, without centering)
- EMOJI (α = 1, proposed)

For each seed, the script plots training loss, validation loss, training perplexity, and validation perplexity against epoch. Thus, it can generate up to 12 plots in total: four metrics for each of the three seeds.

#### Run the per-seed plotting script

From the project root, run:

```powershell
python fairseq/plot_wikitext2_gpt_metrics_per_seed.py
```

The script first exports the CSV files and then opens the available plot previews. Since `SHOW_PREVIEW = True` and `WAIT_FOR_PREVIEW_CLOSE = True`, close each preview window to save its SVG and PNG files and proceed to the next plot. If a metric has no available data for a seed, that plot is skipped.

#### Per-seed plot files

All plot files are written to:

```text
checkpoints/wikitext2_gpt_plots/
```

For each available metric and seed, the script saves both an SVG and a PNG file at 300 DPI. Filenames include the metric, seed, and run timestamp, for example:

```text
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.svg
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.png
```

The four plot filename prefixes are:

- `train_loss_vs_epoch`
- `valid_loss_vs_epoch`
- `train_ppl_vs_epoch`
- `valid_ppl_vs_epoch`

The per-seed plots compare optimizer curves directly within each seed; they do not show a mean curve or standard-deviation band across seeds.

#### CSV files

The per-seed script writes three timestamped CSV files to `checkpoints/wikitext2_gpt_plots/`:

```text
wikitext2_gpt_epochwise_metrics_<RUN_TIMESTAMP>.csv
wikitext2_gpt_final10_summary_<RUN_TIMESTAMP>.csv
wikitext2_gpt_table_final10_<RUN_TIMESTAMP>.csv
```

- **Epoch-wise metrics:** contains the available epoch-level values for every optimizer, seed, and metric, along with the cross-seed mean, sample standard deviation, and number of seeds contributing to each epoch-level statistic.
- **Final-10 summary:** selects the last 10 available epoch values separately for each optimizer, seed, and metric. It reports each seed's mean and sample standard deviation over the selected epochs, followed by the mean and sample standard deviation of the available seed-level means. If fewer than 10 epochs are available for a run, the summary uses the available epochs.
- **Fixed final-10 table:** averages epochs 41–50 separately within each seed. Cross-seed mean and sample standard deviation are reported only when all ten epochs are available for all three seeds for that optimizer and metric; otherwise, the aggregate fields are left blank and a warning is printed.

The final-10 summary and the fixed final-10 table are therefore not necessarily identical when logs have missing epochs: the summary uses the last 10 available epochs, whereas the table specifically uses epochs 41–50.

### EMOJI naming convention

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, every reference to EMOJI in these experiments denotes the **proposed configuration with α = 1**, which uses the centered gradient prediction error.

The value α is not an EMOJI hyperparameter. It is introduced exclusively for the ablation study to represent whether the centering operation is enabled or disabled. Consequently, labels such as `EMOJI`, `Emoji (Proposed)`, and `EMOJI (α = 1, Proposed)` refer to the same proposed configuration.

## Experimental configuration

The experiments use Fairseq's `transformer_lm_gpt` architecture for autoregressive language modeling on WikiText-2.

The shared model and training configuration is summarized below.

| Setting | Value |
|---|---|
| Dataset | WikiText-2 |
| Model architecture | `transformer_lm_gpt` |
| Transformer layers | 4 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Feed-forward network dimension | 512 |
| Dropout | 0.1 |
| Attention dropout | 0.1 |
| Random seeds | 1993, 1402, 2025 |
| Learning-rate scheduler | `inverse_sqrt` |
| Warm-up updates | 1,000 |
| Tokens per sample | 128 |
| Maximum tokens per batch | 1,024 |
| Update frequency | 4 |

The optimizer-specific learning rates and weight decay values used in the comprehensive and four-optimizer comparisons are:

| Optimizer configuration | Learning rate | Weight decay |
|---|---:|---:|
| AdamW | `5e-4` | `0.01` |
| AdaBelief | `5e-4` | `0.01` |
| Lion | `5e-5` | `0.1` |
| EMOJI (Proposed; α = 1) | `5e-4` | `0.01` |
| EMOJI (α = 0, Without Centering) | `5e-4` | `0.01` |

The ablation study compares AdaBelief and the two EMOJI configurations using the settings specified in its experiment configuration.

## Metrics and aggregation across seeds

The scripts extract four metrics from the Fairseq JSON-formatted training logs:

- `train_loss` — training loss.
- `valid_loss` — validation loss.
- `train_ppl` — training perplexity.
- `valid_ppl` — validation perplexity.

### Epoch-wise statistics

For each optimizer and metric, the aggregate scripts calculate the mean and sample standard deviation across the available seeds at each epoch.

The aggregate plots display the epoch-wise mean as a curve, with a shaded region representing one sample standard deviation above and below the mean.

These plots allow the training and validation trajectories of the optimizer configurations to be compared. If a log or metric is unavailable, the scripts process the available data and report missing log files or unavailable metrics.

### Final-10-epoch summary

The final-epoch summary CSV calculates a representative value for each optimizer, seed, and metric by averaging the **final 10 available epoch values separately for each seed**.

It also records the sample standard deviation within those selected epochs. The resulting seed-level means are then aggregated across the available seeds, producing the mean and sample standard deviation across seeds.

This preserves individual-run results while providing aggregate statistics for each optimizer configuration.

### Compact final-10-epoch comparison

The compact table CSV specifically uses **epochs 41–50**.

For each optimizer and metric, it reports the mean over these ten epochs separately for seeds 1993, 1402, and 2025. It then calculates the mean and sample standard deviation across the three seed-level means.

Aggregate statistics are reported only when all ten required epochs are available for all three seeds. Otherwise, the corresponding aggregate fields remain empty.

### Per-seed plotting

In addition to the aggregate plots, `plot_wikitext2_gpt_metrics_per_seed.py` generates separate plots for each seed. Each per-seed plot compares the five optimizer configurations using the same seed and does not average across seeds or display cross-seed standard-deviation bands. For each seed, the script plots training loss, validation loss, training perplexity, and validation perplexity against epoch, producing up to 12 plots in total.

## Generated files

The aggregate scripts save their outputs to the following directory, relative to the project root identified by the script:

`checkpoints/wikitext2_gpt_plots/`

The per-seed plotting script writes its plots and CSV exports to the same directory.

Each analysis script generates three CSV files. The aggregate scripts also generate four SVG figures and four PNG figures when all four metrics have data available for plotting. The per-seed script generates up to 12 per-seed SVG and PNG figures, four metrics for each of the three seeds, when the corresponding data are available.

The output filenames include a run timestamp in the `YYYYMMDD_HHMMSS` format. In the filename patterns below, `{RUN_TIMESTAMP}` represents the actual timestamp generated when the script starts.

### CSV files

| Exact filename pattern | Description |
|---|---|
| `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Contains epoch-wise metric values for each optimizer, seed, and metric. Each row also includes the cross-seed mean, sample standard deviation, and number of available seeds for the corresponding optimizer–metric–epoch combination. |
| `wikitext2_gpt_final10_summary_{RUN_TIMESTAMP}.csv` | Contains the mean and sample standard deviation over the final 10 available epochs for each individual optimizer–seed–metric combination. It also includes `MEAN_ACROSS_SEEDS` rows containing the mean and sample standard deviation across the available seed-level means. If fewer than 10 epochs are available for a run, the summary uses the available epochs. |
| `wikitext2_gpt_table_final10_{RUN_TIMESTAMP}.csv` | Provides a compact table with one row per optimizer. For each metric, it includes individual seed means over epochs 41–50, the mean and sample standard deviation across seeds, the epochs used, and the number of included seeds. Aggregate statistics require complete data for all three seeds and all ten specified epochs; otherwise, the aggregate fields are left blank and a warning is printed. |

The per-seed plotting script writes the same three timestamped CSV files to `checkpoints/wikitext2_gpt_plots/`:

```text
wikitext2_gpt_epochwise_metrics_<RUN_TIMESTAMP>.csv
wikitext2_gpt_final10_summary_<RUN_TIMESTAMP>.csv
wikitext2_gpt_table_final10_<RUN_TIMESTAMP>.csv
```

The three CSV files serve complementary purposes:

- **Epoch-wise metrics:** detailed numerical data for examining training and validation trajectories and comparing individual seeds.
- **Final-10-epoch summary:** per-seed statistics over the final 10 available epochs, together with aggregate statistics across seeds.
- **Compact final-10-epoch table:** a concise comparison based specifically on epochs 41–50.

The final-10 summary and the fixed final-10 table are therefore not necessarily identical when logs have missing epochs: the summary uses the last 10 available epochs, whereas the table specifically uses epochs 41–50.

The CSV exports can be opened in spreadsheet software or processed further with Python for additional analysis.

### Comparing individual seeds in the CSV files

The CSV files support both **optimizer comparisons within the same seed** and **seed comparisons for the same optimizer**.

| Comparison objective | CSV file | How to use it |
|---|---|---|
| Compare different optimizers for the same seed at each epoch | `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Filter by a fixed `seed` and `metric`, then compare the `value` column across optimizers at matching epochs. |
| Compare different seeds for the same optimizer across epochs | `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Filter by a fixed `optimizer` and `metric`, then compare the records for seeds 1993, 1402, and 2025. |
| Compare optimizers using one representative value per seed over the final 10 available epochs | `wikitext2_gpt_final10_summary_{RUN_TIMESTAMP}.csv` | Filter by a fixed `seed` and `metric`, then compare `mean_over_last_epochs` across optimizers. |
| Compare seed-level final-10 means for different optimizers in a compact table | `wikitext2_gpt_table_final10_{RUN_TIMESTAMP}.csv` | Compare columns such as `valid_ppl_seed_1993_last10_mean`, `valid_ppl_seed_1402_last10_mean`, and `valid_ppl_seed_2025_last10_mean` across optimizer rows. |

For a **seed-by-seed comparison between optimizers**, use the same seed for every optimizer. For example, compare AdamW, AdaBelief, Lion, and EMOJI for seed `1993`, then repeat for seeds `1402` and `2025`.

For a **comparison of variability across seeds**, keep the optimizer fixed and compare its results for seeds `1993`, `1402`, and `2025`.

In the final-10-epoch summary, rows labeled `MEAN_ACROSS_SEEDS` contain aggregate statistics rather than individual-seed results. In the compact table, each seed's final-10 mean is stored in its own column.

### SVG and PNG figures

The aggregate scripts generate plots for four metrics.

| Metric | SVG filename | PNG filename |
|---|---|---|
| Training loss | `train_loss_vs_epoch_{RUN_TIMESTAMP}.svg` | `train_loss_vs_epoch_{RUN_TIMESTAMP}.png` |
| Validation loss | `valid_loss_vs_epoch_{RUN_TIMESTAMP}.svg` | `valid_loss_vs_epoch_{RUN_TIMESTAMP}.png` |
| Training perplexity | `train_ppl_vs_epoch_{RUN_TIMESTAMP}.svg` | `train_ppl_vs_epoch_{RUN_TIMESTAMP}.png` |
| Validation perplexity | `valid_ppl_vs_epoch_{RUN_TIMESTAMP}.svg` | `valid_ppl_vs_epoch_{RUN_TIMESTAMP}.png` |

Each aggregate figure displays the epoch-wise mean metric values for the active optimizer configurations. The shaded region represents plus or minus one sample standard deviation across the available seeds.

- **SVG:** a scalable vector format suitable for research papers, publications, and documentation.
- **PNG:** a raster format convenient for previews, presentations, and sharing.

With the current configuration, each plot is displayed in a preview window and saved in both formats when the window is closed. The CSV files are exported before the plotting windows open. If no data are available for a metric, the corresponding figure is not generated.

#### Per-seed figures

The per-seed script saves both an SVG and a PNG file at 300 DPI for each available metric and seed. Filenames include the metric, seed, and run timestamp, for example:

```text
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.svg
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.png
```

The four plot filename prefixes are:

- `train_loss_vs_epoch`
- `valid_loss_vs_epoch`
- `train_ppl_vs_epoch`
- `valid_ppl_vs_epoch`

The per-seed plots compare optimizer curves directly within each seed; they do not show a mean curve or standard-deviation band across seeds.

## TensorBoard logging

TensorBoard logging was enabled in the Fairseq training commands for the WikiText-2 GPT-style Transformer experiments using the `--tensorboard-logdir` argument.

Each training run is associated with a particular optimizer configuration and random seed. The TensorBoard event files are written to the directory specified by that run's `--tensorboard-logdir` argument. When the run directories are organized under the common `checkpoints` directory, TensorBoard can display the logged runs together for interactive comparison.

To launch TensorBoard from the appropriate project environment, run:

```bash
tensorboard --logdir checkpoints
```

Then open the local URL printed in the terminal, typically:

`http://localhost:6006`

The directory supplied to `--logdir` must contain the TensorBoard event files. If the event files are stored in a different location, replace `checkpoints` with the appropriate directory.

### Comparing individual seeds in TensorBoard

TensorBoard allows users to compare **different optimizers using the same seed** and **different seeds using the same optimizer**, provided the corresponding runs have logged event files under the selected log directory.

- **Compare optimizers for the same seed:** select the runs for the desired seed across the optimizer configurations. For example, compare AdamW/seed 1993, AdaBelief/seed 1993, Lion/seed 1993, and EMOJI/seed 1993. This helps compare their recorded training and validation trajectories under the same seed.
- **Compare seeds for the same optimizer:** select the runs for seeds 1993, 1402, and 2025 for one optimizer. This helps inspect how the recorded trajectory varies across random seeds.
- **Repeat for other seeds and optimizers:** use the same procedure for the remaining seed values and optimizer configurations. In the ablation study, compare the three configurations included in that experiment.

For a fair seed-by-seed comparison, select runs with the **same seed across all optimizers**. To inspect seed-to-seed variability, select the **same optimizer across the three seeds**.

The available metrics and run names depend on the event files generated during training and the directory structure used by `--tensorboard-logdir`.

TensorBoard complements the CSV exports and figures by providing an interactive view of the recorded training metrics, while the CSV files provide numerical values for independent analysis.

## Running the analysis scripts

Run the desired script from the configured project environment. The following paths are relative to the project root.

For the comprehensive five-configuration optimizer comparison:

```bash
python checkpoints/plot_wiki2_gpt_abl_all.py
```

For the sequential ablation study:

```bash
python checkpoints/plot_wiki2_gpt_abl.py
```

For the four-optimizer comparison:

```bash
python checkpoints/plot_wiki2_gpt.py
```

For the per-seed plotting and CSV export script:

```powershell
python fairseq/plot_wikitext2_gpt_metrics_per_seed.py
```

The scripts expect their corresponding training logs to exist under the configured checkpoint directory structure. Missing log files are reported, and the scripts continue processing the available runs.

The generated CSV files are written before the plotting windows open. Close each plot window to save its SVG and PNG figures. For the per-seed script, close each preview window to save its SVG and PNG files and proceed to the next plot. If a metric has no available data for a seed, that plot is skipped.

## Summary

The four scripts provide complementary analyses of EMOJI on WikiText-2 using a GPT-style Transformer.

| Script | Relative path | Purpose | Active optimizer configurations |
|---|---|---|---|
| `plot_wiki2_gpt_abl_all.py` | `checkpoints\plot_wiki2_gpt_abl_all.py` | Comprehensive optimizer comparison | AdamW, AdaBelief, Lion, EMOJI (Proposed; α = 1), and EMOJI (α = 0, Without Centering) |
| `plot_wiki2_gpt_abl.py` | `checkpoints\plot_wiki2_gpt_abl.py` | Sequential ablation study | AdaBelief, EMOJI (α = 0, Without Centering), and EMOJI (α = 1, Proposed) |
| `plot_wiki2_gpt.py` | `checkpoints\plot_wiki2_gpt.py` | Four-optimizer comparison | AdamW, AdaBelief, Lion, and EMOJI (Proposed; α = 1) |
| `plot_wikitext2_gpt_metrics_per_seed.py` | `fairseq/plot_wikitext2_gpt_metrics_per_seed.py` | Per-seed plotting and CSV exports | AdamW, AdaBelief, Lion, EMOJI (α = 0, Without Centering), and EMOJI (α = 1, Proposed) |

Each analysis script exports three CSV files containing epoch-wise metrics, final-10-epoch summaries, and a compact table based on epochs 41–50. When the required metric data are available, the aggregate scripts also generate SVG and PNG figures for the four training and validation metrics. The per-seed script generates up to 12 per-seed SVG and PNG figures, four metrics for each of the three seeds.

Together with the TensorBoard logs recorded during Fairseq training, these outputs support detailed examination of optimizer behavior, seed-level comparisons, and reproducible reporting of the WikiText-2 experiments.

## Running the WikiText-2 GPT-style Transformer experiments

**Training script:** `run_wiki_tr_gpt_loop.ps1`

**Relative path:** `fairseq\run_wiki_tr_gpt_loop.ps1`

The PowerShell script `run_wiki_tr_gpt_loop.ps1` automates the WikiText-2 training experiments using Fairseq's GPT-style Transformer architecture. It launches the training runs sequentially, records the training logs, saves checkpoints, and configures TensorBoard logging separately for each optimizer and random seed.

Unlike the plotting scripts described above, this script **runs the training experiments** rather than analyzing existing logs.

### Experiment configurations and execution order

The training script includes **15 runs**:

- **12 standard optimizer runs:** AdamW, EMOJI (Proposed; α = 1), AdaBelief, and Lion, evaluated using three random seeds.
- **3 additional ablation runs:** EMOJI (α = 0, Without Centering), evaluated using the same three seeds.

The standard runs follow this order:

| Random seed | Execution order |
|---|---|
| `1993` | AdamW → EMOJI → AdaBelief → Lion |
| `1402` | AdamW → EMOJI → AdaBelief → Lion |
| `2025` | AdamW → EMOJI → AdaBelief → Lion |

After the 12 standard runs, the script executes the three additional EMOJI no-centering runs for seeds `1993`, `1402`, and `2025`.

Each run is configured for a maximum of 50 epochs.

### Model and training configuration

All runs use the following GPT-style Transformer configuration:

| Setting | Value |
|---|---|
| Dataset | WikiText-2 |
| Architecture | `transformer_lm_gpt` |
| Transformer layers | 4 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Feed-forward network dimension | 512 |
| Activation function | `gelu` |
| Dropout | 0.1 |
| Attention dropout | 0.1 |
| Learning-rate scheduler | `inverse_sqrt` |
| Warm-up updates | 1,000 |
| Warm-up initial learning rate | `1e-7` |
| Tokens per sample | 128 |
| Maximum tokens per batch | 1,024 |
| Update frequency | 4 |
| Maximum epochs | 50 |
| Random seeds | 1993, 1402, 2025 |
| Fixed validation seed | Same as the run's training seed |
| Gradient clipping norm | 0.0 |
| Number of data-loader workers | 0 |

### Optimizer-specific settings

The standard optimizer configurations use the following hyperparameters:

| Optimizer | Learning rate | Weight decay | Betas | Epsilon |
|---|---:|---:|---|---:|
| AdamW | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| EMOJI (Proposed; α = 1) | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| AdaBelief | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| Lion | `5e-5` | `0.1` | `(0.95, 0.98)` | Not applicable |

The additional ablation runs use the following configuration:

| Setting | Value |
|---|---|
| Optimizer | `emojiabl` |
| Centering configuration | α = 0 |
| Learning rate | `5e-4` |
| Weight decay | `0.01` |
| Betas | `(0.9, 0.98)` |
| Epsilon | `1e-8` |
| Random seeds | 1993, 1402, 2025 |

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, EMOJI refers to the proposed configuration with α = 1. The value α is not an EMOJI hyperparameter; it is used only in the ablation study to indicate whether gradient prediction error centering is enabled or disabled.

### Run directories and saved files

Each training run has a separate output directory under `checkpoints/`. The directory structure distinguishes optimizer configurations, their training settings, and individual random seeds.

For the four standard optimizers, the general directory structure is:

```text
checkpoints/
└── wikitext-2_transformer_lm_gpt_<optimizer>/
    └── Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD<weight_decay>_WU1000_LR<lr>_Schedinverse_sqrt_TPS128_MT1024_UF4/
        └── seed_<seed>/
            ├── train.log
            ├── tensorboard/
            └── Fairseq checkpoint files
```

The additional EMOJI no-centering runs are stored separately:

```text
checkpoints/
└── wikitext-2_transformer_lm_gpt_emojiabl/
    └── Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/
        ├── seed_1993/
        ├── seed_1402/
        └── seed_2025/
```

Each seed-specific directory contains the corresponding `train.log`, a `tensorboard/` directory, and the checkpoint files produced by Fairseq.

The training logs use JSON-formatted output and record the metrics needed by the plotting scripts. The separate directory structure also allows each training run to be inspected independently.

### TensorBoard logging and seed-by-seed comparison

The script passes `--tensorboard-logdir` to Fairseq, assigning a separate TensorBoard event directory to each optimizer–seed combination.

After training, launch TensorBoard from the project environment:

```bash
tensorboard --logdir checkpoints
```

Then open the local URL printed in the terminal, typically:

`http://localhost:6006`

Because the event directories are nested under `checkpoints/`, TensorBoard can display the runs together when their event files are present under that directory.

TensorBoard supports two complementary comparisons:

- **Compare optimizers using the same seed:** select AdamW, EMOJI, AdaBelief, and Lion for seed `1993`, for example, and compare their recorded training and validation trajectories. Repeat for seeds `1402` and `2025`. In the ablation study, include the corresponding EMOJI (α = 0) run.
- **Compare seeds using the same optimizer:** select seeds `1993`, `1402`, and `2025` for one optimizer to inspect how its training and validation trajectories vary across random seeds.

For a seed-matched comparison, use the same seed across all optimizer configurations. For a seed-variability comparison, keep the optimizer fixed and inspect all three seeds.

The event files provide the interactive training curves, while the plotting scripts generate the CSV summaries and publication-ready figures from the recorded `train.log` files.

#### Filtering runs and metrics

To compare validation metrics across all five optimizer configurations for a specific random seed, use the following regular expression in TensorBoard:

```regex
^(?=.*\*seed_1993)(?=.\**\bvalid\b).\*$
```

This expression is intended to select the validation metrics associated with seed `1993` across all five configurations:

- AdamW
- AdaBelief
- Lion
- EMOJI (α = 0, without centering)
- EMOJI (α = 1, proposed)

To perform the same comparison for another seed, replace `1993` in the expression with `1402` or `2025`, respectively.

This allows a seed-matched comparison of the validation metrics across all five configurations while keeping the random seed consistent.

### Running the training script

Open PowerShell in the Fairseq project directory containing `data-bin/wikitext-2`, make sure the configured Python environment and Fairseq installation are available, and run:

```powershell
.\run_wiki_tr_gpt_loop.ps1
```

The relative path from the project root is `fairseq\run_wiki_tr_gpt_loop.ps1`.

The script launches the standard optimizer runs in the specified order and subsequently executes the three additional EMOJI no-centering runs.

The 12 standard runs have exit-code checks and are recorded in the `$failedRuns` summary. In the supplied script, the three standalone `emojiabl` commands are outside that loop and are not included in its final failure summary.

The script uses the existing WikiText-2 binarized dataset and does not perform data preprocessing. It saves each run's logs and checkpoints to its corresponding output directory, allowing the resulting experiments to be analyzed with the plotting scripts documented above.

---

# Transformer-based machine translation on IWSLT14

We compared four optimizer implementations—AdamW, AdaBelief, EMOJI, and Lion—on the IWSLT14 German-to-English machine translation task using Fairseq. All runs used the `transformer_iwslt_de_en` architecture and the same dataset and shared training configuration. The experiment protocol used the same three random seeds (1993, 1402, and 2025) across optimizer configurations.

### Script Locations (relative to repository root)

```text
├── fairseq/
│   ├── run_IWSLT_GE_2_EN.ps1
│   ├── run_IWSLT_Sensitivity.ps1
│   └── run_IWSLT_Sensitivity_1seed.ps1
└── checkpoints/
    ├── plot_iwslt14_epsilon_sensitivity_combined.py
    ├── plot_iwslt14_train_metrics_mean_var_ablation.py
    ├── plot_iwslt14_train_metrics_mean_var_all.py
    ├── plot_iwslt14_train_metrics_mean_var_per_seed.py
    ├── plot_iwslt14_train_metrics_mean_var.py
    └── plot_iwslt14_train_metrics_sensitivity2.py
```

## Model architecture and training configuration

The models used Fairseq's `transformer_iwslt_de_en` architecture. The saved configuration inspected for the EMOJI run confirms six encoder layers and six decoder layers, 512-dimensional embeddings, feed-forward networks of dimension 1,024, and four attention heads in both the encoder and decoder. The model used ReLU activation, dropout of 0.3, zero attention and activation dropout, post-layer normalization, non-learned positional embeddings, and shared decoder input/output embeddings.

**Table 1. Model architecture and configuration.**

| Component | Configuration |
|---|---|
| Task | German-to-English machine translation |
| Dataset | IWSLT14 German-to-English |
| Fairseq task | `translation` |
| Fairseq architecture | `transformer_iwslt_de_en` |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Encoder embedding dimension | 512 |
| Decoder embedding dimension | 512 |
| Encoder attention heads | 4 |
| Decoder attention heads | 4 |
| Attention head dimension | 128 |
| Encoder feed-forward dimension | 1,024 |
| Decoder feed-forward dimension | 1,024 |
| Activation function | ReLU |
| Decoder input/output embeddings | Shared |
| Encoder layer normalization | Post-LayerNorm (Pre-LN disabled) |
| Decoder layer normalization | Post-LayerNorm (Pre-LN disabled) |
| Embedding layer normalization | Disabled |
| Learned positional embeddings | Disabled |
| Maximum source positions | 1,024 tokens |
| Maximum target positions | 1,024 tokens |
| Model dropout | 0.3 |
| Attention dropout | 0.0 |
| Activation dropout | 0.0 |

The shared training and evaluation configuration is summarized in Table 2. The listed architecture details are present in the inspected EMOJI checkpoint configuration; the five supplied commands use the same architecture name and common command-line settings.

**Table 2. Shared training configuration.**

| Training parameter | Value |
|---|---|
| Maximum epochs | 50 |
| Maximum tokens per batch | 4,096 |
| Gradient accumulation (`update_freq`) | 1 (Fairseq configuration) |
| Learning-rate scheduler | Inverse square root (`inverse_sqrt`) |
| Warm-up updates | 4,000 |
| Gradient clipping (`clip_norm`) | 0.0 (disabled) |
| Training objective | Label-smoothed cross-entropy |
| Label-smoothing factor | 0.1 |
| Logging format | JSON |
| Logging interval | 100 updates (stored checkpoint configuration) |
| Data-loader workers | 1 (stored checkpoint configuration) |
| Validation frequency | Every epoch (stored checkpoint configuration) |
| Maximum validation tokens per batch | 4,096 (stored checkpoint configuration) |
| Random seeds in the experiment protocol | 1993, 1402, 2025 |


All command examples use the inverse-square-root learning-rate scheduler with 4,000 warm-up updates, disabled gradient clipping, dropout of 0.3, label-smoothed cross-entropy with a smoothing factor of 0.1, and a maximum of 4,096 tokens per batch. Optimizer-specific learning rates, beta coefficients, and weight decay are listed separately in Table 3.

## Optimizer configurations

AdamW, AdaBelief, and EMOJI were evaluated using the same learning rate of `5e-4`, weight decay of `0.0001`, and moment coefficients β₁ = 0.9 and β₂ = 0.98. The choice of these coefficients follows the original Transformer paper, *Attention Is All You Need*, which used Adam with β₁ = 0.9, β₂ = 0.98. In our experiments, we retained the same beta coefficients and used ε = 1e-8, together with a common inverse-square-root learning-rate schedule with 4,000 warm-up updates. Applying the same settings to AdamW, AdaBelief, and EMOJI provides a controlled comparison of their optimization behavior under a shared configuration.

**Table 3. Optimizer-specific hyperparameters.**


| Optimizer | Learning rate | β₁ | β₂ | Epsilon | Weight decay |
|---|---|---|---|---|---|
| AdamW | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| AdaBelief | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| EMOJI | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| Lion | `5e-5` | 0.95 | 0.98 | Not applicable | 0.001 |

Lion was configured according to recommendations in the Lion literature. Its learning rate was set to one-tenth of the AdamW learning rate, while its weight decay was increased tenfold. This adjustment is motivated by Lion's sign-based update, which can produce updates with larger norms than AdamW's and therefore commonly calls for a smaller learning rate and a larger weight-decay coefficient. The coefficients β₁ = 0.95 and β₂ = 0.98 are also consistent with settings investigated for Lion in language-modeling experiments. These values are optimizer-specific recommendations rather than the default beta values of Lion.

The Lion configuration therefore follows the recommended relative scaling of learning rate and weight decay, rather than enforcing identical hyperparameters across all optimizers. Consequently, the comparison evaluates the four methods under the specified configurations, not under a strict requirement that every optimizer use identical optimizer-specific hyperparameters.

## Reproducibility

Each optimizer was evaluated using the same three random seeds, with all other shared model and training settings held constant. This shared-seed protocol helps control for variability due to random initialization and training stochasticity when comparing results across optimizers.

**Harmonized continuation (aligned with the section above):**

## Statistical Convention

Throughout all analyses, the representative value for each seed is the **mean of the final 10 validation epochs**.  

When three seeds are available, the reported mean and sample standard deviation (ddof = 1) are computed **across these three seed-level final-10 means**. Raw per-epoch values are never averaged across seeds. This rule is applied uniformly to every table, figure, and CSV produced by the plotting scripts.

## EMOJI Naming Convention

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, every reference to **EMOJI** denotes the proposed configuration with centering enabled (α = 1).  

The parameter α is introduced solely for the ablation study and is **not** a tunable hyperparameter of the proposed optimizer. Consequently:

- `iwslt14_emoji` = EMOJI (α = 1, Proposed)  
- `iwslt14_emojiabl` = EMOJI (α = 0, Without Centering)

## Plotting & Analysis Scripts

All scripts reside in the `checkpoints/` directory and write outputs to `checkpoints_of_seeds/plots/`. Input logs are Fairseq JSON-line files located at `checkpoints_of_seeds/<seed>/<run_folder>/train.log`.

### 1. `plot_iwslt14_epsilon_sensitivity_combined.py`

Produces ε-sensitivity analysis of EMOJI (α = 1) versus AdaBelief for `valid_bleu`, `valid_ppl`, and `valid_loss`.

- Single-seed curves (seed 1993) over six ε values: 1e-24, 1e-16, 1e-12, 1e-10, 1e-8, 1e-6  
- Three-seed mean ± sample STD over ε ∈ {1e-16, 1e-12, 1e-8}  
- Combined view (mean markers + single-seed dotted curves)

Outputs (per metric): three SVG figures and two CSV files following the final-10 statistical convention.

### 2. `plot_iwslt14_train_metrics_mean_var_ablation.py`

Epoch-wise mean ± 1 sample STD across the three seeds for the ablation comparison:

- EMOJI (α = 1, Proposed)  
- AdaBelief  
- EMOJI (α = 0, Without Centering)

Metrics: `train_loss`, `valid_loss`, `valid_bleu`, `valid_ppl`.

### 3. `plot_iwslt14_train_metrics_mean_var_all.py`

Epoch-wise mean ± 1 sample STD across five configurations:

- EMOJI (α = 1, Proposed)  
- AdaBelief  
- AdamW  
- Lion  
- EMOJI (α = 0, Without Centering)

Files are timestamped and saved as both SVG and PNG on figure close.

### 4. `plot_iwslt14_train_metrics_mean_var_per_seed.py`

Per-seed epoch curves (no cross-seed averaging) for the same five configurations. Produces 12 figures (3 seeds × 4 metrics), each saved as SVG + PNG.

### 5. `plot_iwslt14_train_metrics_mean_var.py`

Bulk CSV export covering nine runs (base optimizers + selected ε variants). Main outputs:

- `iwslt14_all_metrics_mean_std_last10.csv` — cross-seed mean ± sample STD of final-10 means  
- `iwslt14_all_metrics_last10_per_seed.csv` — per-seed final-10 means  
- Epoch-wise mean/STD and raw per-seed comparative CSVs

### 6. `plot_iwslt14_train_metrics_sensitivity2.py`

Presentation-oriented ε-sensitivity plots:

- Four three-seed mean ± STD epoch plots  
- Four single-seed (seed 1993) epoch plots  
- Standalone legend  
- Two CSV files with final-10 statistics (single-seed and three-seed)

All figures use 300 dpi, tight bounding boxes, and follow the same final-10 / cross-seed statistical protocol.

## Experiment Launchers

### Base launcher — `run_IWSLT_GE_2_EN.ps1`

Launches the core 15 runs (3 seeds × 5 configurations):

- `iwslt14_adamw`  
- `iwslt14_emoji` (α = 1)  
- `iwslt14_adabelief`  
- `iwslt14_emojiabl` (α = 0)  
- `iwslt14_lion`

All runs share the architecture and training settings listed in Tables 1–2 and the optimizer hyperparameters of Table 3. Lion uses the distinct learning rate, weight decay, and β values shown in Table 3.

### ε-sensitivity launchers

- `run_IWSLT_Sensivity.ps1` — three-seed sweep for ε ∈ {1e-12, 1e-16, 1e-8 (default)}  
- `run_IWSLT_Sensivity_1seed.ps1` — single-seed (1993) sweep for ε ∈ {1e-24, 1e-10, 1e-6}

These complete the six-ε single-seed and three-ε multi-seed trajectories consumed by scripts 1 and 6.

Default ε = 1e-8 is obtained by omitting the `--emoji-eps` / `--adabelief-eps` flag. All EMOJI ε-variants use the proposed configuration (α = 1).

---

# CIFAR-10 Image Classification with DenseNet121, ResNet20, and ResNet34

This document describes the experimental setup, analysis scripts, and launchers for comparing optimizers on CIFAR-10 image classification. The experiments cover three architectures (DenseNet121, ResNet20, ResNet34) and evaluate both a full multi-optimizer comparison and a focused ablation of the EMOJI centering mechanism.

All scripts reside under the repository root `emoji_optimizer/classification_cifar10_resnet20`. Analysis scripts live at the relative paths listed below; training is launched via the PowerShell scripts in `schedulers/`.

### Script Locations (relative to repository root)

```text
emoji_optimizer/classification_cifar10_resnet20/
├── final_densenet121alloptimsfig.py.py
├── final_resnet20alloptimsfig.py.py
├── final_resnet34alloptimsfig.py
├── ablation_meanstd95densnet121.py
├── ablation_meanstd95resnet20.py
├── ablation_meanstd95resnet34.py
└── schedulers/
    ├── run_densenet121.ps1
    ├── run_resnet20.ps1
    └── run_resnet34.ps1
```

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

---

# ALBERT-base on GLUE (CoLA, MRPC, STS-B)

This document describes the experimental setup, analysis scripts, and launchers for evaluating the proposed EMOJI optimizer against AdamW, AdaBelief, and Lion on three GLUE tasks using ALBERT-base.

All analysis scripts and the training launcher reside under the corresponding experiment directory. TensorBoard event logs are stored in the `logs/` directory. The analysis scripts read these logs to compute means and standard deviations.

### Script Locations (relative to repository root)

```text
albert_finetuning/
├── final_cola_alloptim.py
├── final_mrpc_alloptim.py
├── final_sts_alloptim.py
└── run_ALBERT_FN.ps1
```

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

---

# ImageNet-1K Classification with ResNet-18

This document describes the experimental setup for evaluating the proposed EMOJI optimizer against AdaBelief on ImageNet-1K using the ResNet-18 architecture.

### Script Locations (relative to repository root)

```text
imagenet/
├── main_when_copy_2.py
├── final_resnet18alloptimsfig.py
├── run.ps1
└── imagenet_evaluation.ps1
```

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
**Updated section for the Markdown:**

---

### Analysis Script

`final_resnet18alloptimsfig.py`

- Reads the per-epoch CSV logs produced during training.
- Plots Top-1 and Top-5 validation accuracy curves.
- Computes the mean of the last 10 epochs for both Top-1 and Top-5 accuracy.
- Saves summary CSVs and SVG figures.

**Per-epoch training logs (CSV):**

- AdaBelief:  
  `model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.99901-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False.csv`

- EMOJI (Proposed):  
  `model-resnet18-optimizer-emoji-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-emoji_eps1e-08-emoji_beta10.9-emoji_beta20.999-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False.csv`

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

