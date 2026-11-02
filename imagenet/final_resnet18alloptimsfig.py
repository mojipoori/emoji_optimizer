
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =============================================================================
# USER CONFIGURATION
# =============================================================================

MODEL_NAME = "ResNet18"
TASK = "Image Classification"
DATASET = "ImageNet-1K"

# Number of final epochs used for the final CSV summary.
LAST_N_EPOCHS = 10

# Figure settings.
FIGSIZE = (8, 5)
SAVE_DPI = 300

# Exact title for ImageNet figures.
PLOT_TITLE = "Classification on ImageNet-1K using ResNet-18"

# Output directory for summary CSV files.
OUTPUT_DIR = "csv_summaries"


# =============================================================================
# INPUT CSV FILES
# =============================================================================

OPTIMIZERS = {

    'AdaBelief': [

        # Add one CSV path per seed.
        'model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.99901-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False.csv',

    ],

    'Emoji (Proposed)': [

        # Add one CSV path per seed.
        'model-resnet18-optimizer-emoji-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-emoji_eps1e-08-emoji_beta10.9-emoji_beta20.999-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False.csv',

    ],

}


# =============================================================================
# METRICS USED FOR PLOTTING
# =============================================================================

# Expected CSV columns:
#
# epoch,val_acc1,val_acc5
#
# Training metrics are not stored or plotted.

TAGS_TO_PLOT = {

    'Accuracy/validation (Top-1)': 'val_acc1',

    'Accuracy/validation (Top-5)': 'val_acc5',

}


# =============================================================================
# DISPLAY LABELS
# =============================================================================

DISPLAY_LABELS = {

    'Accuracy/validation (Top-1)': 'Top-1 Validation Accuracy (%)',

    'Accuracy/validation (Top-5)': 'Top-5 Validation Accuracy (%)',

}


# =============================================================================
# TOP-1 / TOP-5 COLUMNS FOR FINAL CSV
# =============================================================================

TOP1_COLUMN = 'val_acc1'
TOP5_COLUMN = 'val_acc5'


# =============================================================================
# PLOT COLORS
# =============================================================================

COLORS = {

    'AdaBelief': '#D62728',

    'Emoji (Proposed)': '#1F77B4',

}


# =============================================================================
# CSV LOADING / VALIDATION
# =============================================================================

def read_results_csv(csv_path):

    """
    Read and validate a results CSV.

    Expected columns:
        epoch
        val_acc1
        val_acc5
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        'epoch',
        'val_acc1',
        'val_acc5',
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing columns in CSV: {missing_columns}\n"
            f"File: {csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Convert required columns to numeric values.
    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce',
        )

    # Remove rows with missing values.
    df = df.dropna(
        subset=required_columns
    )

    # Sort by epoch.
    df = df.sort_values(
        'epoch'
    ).reset_index(
        drop=True
    )

    if df.empty:

        raise ValueError(
            f"No valid data found in CSV: {csv_path}"
        )

    return df


# =============================================================================
# LOAD DATA FOR PLOT
# =============================================================================

def load_optimizer_data(
    csv_paths,
    tag,
):

    all_dfs = []

    col = TAGS_TO_PLOT.get(tag)

    if col is None:

        raise ValueError(
            f"Unknown plotting tag: {tag}"
        )

    for i, fpath in enumerate(csv_paths):

        p = Path(fpath)

        if not p.exists():

            print(
                f"  CSV file not found: {p}"
            )

            continue

        try:

            df = read_results_csv(p)

        except Exception as error:

            print(
                f"  Could not load {p}: {error}"
            )

            continue

        sub = df[
            ['epoch', col]
        ].copy()

        sub.rename(
            columns={
                col: 'value'
            },
            inplace=True,
        )

        sub['seed'] = i + 1

        all_dfs.append(sub)

    if not all_dfs:

        return (
            None,
            None,
            None,
        )

    full = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    # Calculate mean across seeds for each epoch.
    grouped = (
        full
        .groupby('epoch')['value']
        .agg(
            mean='mean',
            n_seeds='count',
        )
        .sort_index()
    )

    epochs = grouped.index.values

    means = grouped['mean'].values

    n_seeds_per_epoch = grouped['n_seeds'].values

    return (
        epochs,
        means,
        n_seeds_per_epoch,
    )


# =============================================================================
# LOAD RAW SEED DATA
# =============================================================================

def load_seed_data(
    csv_paths,
    column_name,
):

    seed_data = []

    for i, fpath in enumerate(csv_paths):

        p = Path(fpath)

        if not p.exists():

            print(
                f"  CSV file not found for "
                f"seed {i + 1}: {p}"
            )

            continue

        try:

            df = read_results_csv(p)

        except Exception as error:

            print(
                f"  Could not load seed {i + 1}: {error}"
            )

            continue

        if column_name not in df.columns:

            print(
                f"  Column '{column_name}' "
                f"not found in {p}"
            )

            continue

        sub = df[
            ['epoch', column_name]
        ].copy()

        sub.rename(
            columns={
                column_name: 'value'
            },
            inplace=True,
        )

        sub = sub.dropna(
            subset=[
                'epoch',
                'value',
            ]
        )

        sub = sub.sort_values(
            'epoch'
        )

        seed_data.append(
            {
                'seed': i + 1,
                'path': p,
                'data': sub,
            }
        )

    return seed_data


# =============================================================================
# CALCULATE MEAN OF LAST N EPOCHS
# =============================================================================

def calculate_last_n_mean(
    csv_paths,
    column_name,
    n=LAST_N_EPOCHS,
):

    seed_data = load_seed_data(
        csv_paths,
        column_name,
    )

    if not seed_data:

        return None

    seed_means = []

    epoch_starts = []
    epoch_ends = []

    print()

    print(
        f"  Last {n} epochs "
        f"for column '{column_name}':"
    )

    for item in seed_data:

        df = item['data']

        if len(df) < n:

            raise ValueError(
                f"\nSeed {item['seed']} has only "
                f"{len(df)} valid epochs, but {n} "
                f"epochs are required.\n"
                f"File: {item['path']}"
            )

        # Select the last N epochs.
        last_n = df.tail(n)

        # Calculate mean for this seed.
        mean_value = float(
            last_n['value'].mean()
        )

        seed_means.append(
            mean_value
        )

        epoch_start = int(
            last_n['epoch'].iloc[0]
        )

        epoch_end = int(
            last_n['epoch'].iloc[-1]
        )

        epoch_starts.append(
            epoch_start
        )

        epoch_ends.append(
            epoch_end
        )

        print(
            f"    Seed {item['seed']}: "
            f"epochs {epoch_start} -> {epoch_end} | "
            f"mean = {mean_value:.6f}"
        )

    # Calculate mean across seeds.
    final_mean = float(
        np.mean(seed_means)
    )

    return {

        'mean_last_n': final_mean,

        'seed_means': seed_means,

        'n_seeds': len(seed_means),

        'epoch_start': min(epoch_starts),

        'epoch_end': max(epoch_ends),

    }


# =============================================================================
# SAVE FINAL TOP-1 / TOP-5 CSV
# =============================================================================

def save_final_top1_top5_csv(
    model_name,
    optimizer_results,
    output_dir=OUTPUT_DIR,
):

    out_path = Path(
        output_dir
    )

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for label, result in optimizer_results.items():

        if result is None:
            continue

        rows.append(
            {

                'optimizer': label,

                f'top1_mean_last_{LAST_N_EPOCHS}_epochs':
                    result['top1_mean_last_n_epochs'],

                f'top5_mean_last_{LAST_N_EPOCHS}_epochs':
                    result['top5_mean_last_n_epochs'],

            }
        )

    if not rows:

        print(
            "  No final Top-1 / Top-5 results available."
        )

        return

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        out_path
        / f"{model_name}_"
          f"top1_top5_last{LAST_N_EPOCHS}_mean.csv"
    )

    # Save final results as CSV.
    df.to_csv(
        csv_path,
        index=False,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL TOP-1 / TOP-5 CSV"
    )

    print(
        "=" * 70
    )

    print(
        df.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved final CSV:"
    )

    print(
        csv_path
    )


# =============================================================================
# SAVE EPOCH-WISE CSV
# =============================================================================

def save_tag_csv(
    model_name,
    tag,
    optimizers_data,
    output_dir=OUTPUT_DIR,
):

    out_path = Path(
        output_dir
    )

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for od in optimizers_data:

        for epoch, mean_value, n_seeds in zip(
            od['epochs'],
            od['mean'],
            od['n_seeds_per_epoch'],
        ):

            rows.append(
                {

                    'epoch': epoch,

                    'optimizer': od['label'],

                    'mean': mean_value,

                    'n_seeds': n_seeds,

                }
            )

    df = pd.DataFrame(
        rows
    )

    safe_tag = tag.replace(
        '/',
        '_',
    )

    fname = (
        out_path
        / f"{model_name}_{safe_tag}_summary.csv"
    )

    df.to_csv(
        fname,
        index=False,
    )

    print(
        f"  Saved epoch-wise summary CSV to {fname}"
    )


# =============================================================================
# MAIN LOOP: FINAL TOP-1 / TOP-5 RESULTS
# =============================================================================

final_results = {}

print()

print(
    "=" * 80
)

print(
    f"FINAL IMAGE-NET RESULTS "
    f"USING LAST {LAST_N_EPOCHS} EPOCHS"
)

print(
    "=" * 80
)


for label, csv_paths in OPTIMIZERS.items():

    if not csv_paths:
        continue

    print()

    print(
        f"Optimizer: {label}"
    )

    # ---------------------------------------------------------------------
    # Top-1
    # ---------------------------------------------------------------------

    print()

    print(
        "Top-1:"
    )

    top1_result = calculate_last_n_mean(
        csv_paths,
        TOP1_COLUMN,
        n=LAST_N_EPOCHS,
    )

    # ---------------------------------------------------------------------
    # Top-5
    # ---------------------------------------------------------------------

    print()

    print(
        "Top-5:"
    )

    top5_result = calculate_last_n_mean(
        csv_paths,
        TOP5_COLUMN,
        n=LAST_N_EPOCHS,
    )

    if (
        top1_result is None
        or top5_result is None
    ):

        final_results[label] = None

        continue

    final_results[label] = {

        'top1_mean_last_n_epochs':
            top1_result['mean_last_n'],

        'top5_mean_last_n_epochs':
            top5_result['mean_last_n'],

    }

    print()

    print(
        f"  FINAL Top-1 mean "
        f"(last {LAST_N_EPOCHS} epochs): "
        f"{top1_result['mean_last_n']:.6f}"
    )

    print(
        f"  FINAL Top-5 mean "
        f"(last {LAST_N_EPOCHS} epochs): "
        f"{top5_result['mean_last_n']:.6f}"
    )


# =============================================================================
# SAVE FINAL TOP-1 / TOP-5 CSV
# =============================================================================

save_final_top1_top5_csv(
    MODEL_NAME,
    final_results,
)


# =============================================================================
# PLOTS
# =============================================================================

for tag, col in TAGS_TO_PLOT.items():

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Processing tag: {tag}"
    )

    print(
        f"CSV column: {col}"
    )

    print(
        f"{'=' * 70}"
    )

    tag_data = []

    # ---------------------------------------------------------------------
    # Load data for plotting.
    # ---------------------------------------------------------------------

    for label, csv_paths in OPTIMIZERS.items():

        print(
            f"\nLoading {label} ..."
        )

        if not csv_paths:
            continue

        (
            epochs,
            mean,
            n_seeds_per_epoch,
        ) = load_optimizer_data(
            csv_paths,
            tag,
        )

        if epochs is None:
            continue

        tag_data.append(
            {

                'label': label,

                'epochs': epochs,

                'mean': mean,

                'n_seeds_per_epoch': n_seeds_per_epoch,

            }
        )

        print(
            f"  Seeds: {len(csv_paths)}"
        )

    if not tag_data:

        print(
            f"No data for tag '{tag}'. "
            f"Skipping."
        )

        continue

    # ---------------------------------------------------------------------
    # Create figure.
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    # ---------------------------------------------------------------------
    # Plot mean across seeds.
    # Every epoch gets a marker.
    # ---------------------------------------------------------------------

    for idx, od in enumerate(tag_data):

        label = od['label']

        color = COLORS.get(
            label,
            f'C{idx}',
        )

        epochs = np.asarray(
            od['epochs']
        )

        mean = np.asarray(
            od['mean']
        )

        ax.plot(
            epochs,
            mean,
            color=color,
            label=label,
            linewidth=2.0,
            alpha=0.85,
            marker='o',
            markersize=1.5,
        )

    # ---------------------------------------------------------------------
    # Axes.
    # ---------------------------------------------------------------------

    ax.set_xlabel(
        'Epoch',
        fontsize=11,
    )

    ax.set_ylabel(
        DISPLAY_LABELS.get(
            tag,
            tag,
        ),
        fontsize=11,
    )

    ax.set_title(
        PLOT_TITLE,
        fontsize=12,
    )

    # ---------------------------------------------------------------------
    # Grid.
    # ---------------------------------------------------------------------

    ax.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.4,
    )

    # ---------------------------------------------------------------------
    # Legend.
    # ---------------------------------------------------------------------

    ax.legend(
        fontsize=8.5,
        frameon=True,
        markerscale=1.0,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.25,
        borderpad=0.3,
        borderaxespad=0.4,
        columnspacing=0.8,
    )

    # ---------------------------------------------------------------------
    # Layout.
    # ---------------------------------------------------------------------

    fig.tight_layout()

    # ---------------------------------------------------------------------
    # Save SVG.
    # ---------------------------------------------------------------------

    figure_filename = (
        f'{MODEL_NAME}_'
        f'{tag.replace("/", "_")}.svg'
    )

    fig.savefig(
        figure_filename,
        format='svg',
        dpi=SAVE_DPI,
        bbox_inches='tight',
    )

    print()

    print(
        f"  Saved figure: "
        f"{figure_filename}"
    )

    # ---------------------------------------------------------------------
    # Display.
    # ---------------------------------------------------------------------

    plt.show()

    # ---------------------------------------------------------------------
    # Close figure.
    # ---------------------------------------------------------------------

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Save epoch-wise CSV.
    # ---------------------------------------------------------------------

    save_tag_csv(
        MODEL_NAME,
        tag,
        tag_data,
    )

    print(
        f"  Finished processing {tag}."
    )


# =============================================================================
# COMPLETION
# =============================================================================

print()

print(
    "=" * 70
)

print(
    "ALL PROCESSING COMPLETED"
)

print(
    "=" * 70
)
