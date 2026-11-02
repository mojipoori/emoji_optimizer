import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


# =============================================================================
# USER CONFIGURATION – edit this section only
# =============================================================================

MODEL_NAME = "DenseNet121"

TASK = "Image Classification"

DATASET = "CIFAR-10"

# Number of final COMMON epochs used for the final summary.
LAST_N_EPOCHS = 30

# Output directory for CSV files.
OUTPUT_DIR = Path("csv_summaries")


# =============================================================================
# OPTIMIZERS
# =============================================================================

OPTIMIZERS = {

    'AdaBelief': [

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-055632-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-062442-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-093547-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-151245-resetFalse',

    ],

    'Emoji (Proposed)': [

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed8-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-091625-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed2025-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-130029-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-163922-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed1402-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-003415-resetFalse',

    ],

    # -------------------------------------------------------------------------
    # NEW OPTIMIZERS
    # -------------------------------------------------------------------------

    'AdaBound': [

        'runs/densenet-adaboundw-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.01-eps1e-08-amsboundFalse-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-adaboundw-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.01-eps1e-08-amsboundFalse-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251022-184131-resetFalse',

        'runs/densenet-adaboundw-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.01-eps1e-08-amsboundFalse-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251022-215959-resetFalse',

        'runs/densenet-adaboundw-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.01-eps1e-08-amsboundFalse-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20260619-040447-resetFalse',

    ],

    'AdamW': [

        'runs/densenet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-201507-resetFalse',

        'runs/densenet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-222402-resetFalse',

        'runs/densenet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-041706-resetFalse',

        'runs/densenet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-130422-resetFalse',

    ],

    'Yogi': [

        'runs/densenet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.01-run0-seed0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.01-run0-seed8-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.01-run0-seed2025-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.01-run0-seed1402-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

    ],

    'Lion': [

        'runs/densenet-lion-lr0.0001-betas0.9-0.999-wdecay0.1-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-lion-lr0.0001-betas0.9-0.999-wdecay0.1-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251021-224223-resetFalse',

        'runs/densenet-lion-lr0.0001-betas0.9-0.999-wdecay0.1-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251022-112922-resetFalse',

        'runs/densenet-lion-lr0.0001-betas0.9-0.999-wdecay0.1-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20260619-012458-resetFalse',

    ],

    'SGDW': [

        'runs/densenet-sgdw-lr0.1-mom0.9-wd0.01-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/densenet-sgdw-lr0.1-mom0.9-wd0.01-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251022-140703-resetFalse',

        'runs/densenet-sgdw-lr0.1-mom0.9-wd0.01-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251022-162412-resetFalse',

        'runs/densenet-sgdw-lr0.1-mom0.9-wd0.01-run0-seed1402-warmupsteplr--linT750-linS0.001-slrGam0.1-resetFalse-resume-20260619-072509',

    ],

}


# =============================================================================
# METRICS
# =============================================================================

TAGS_TO_PLOT = [

    'Accuracy/test',

    'Accuracy/train',

    'Loss/test',

    'Loss/train',

]


# =============================================================================
# COLORS
# =============================================================================

COLORS = {
    'AdaBelief': '#D62728',
    'Emoji (Proposed)': '#1F77B4',
    'AdaBound': '#2CA02C',
    'AdamW': '#9467BD',
    'Yogi': '#8C564B',
    'Lion': '#E377C2',
    'SGDW': '#7F7F7F',
}


# =============================================================================
# EXTRACT SEED NAME
# =============================================================================

def extract_seed_name(
    path_string: str,
) -> str:

    matches = re.findall(
        r"seed(\d+)",
        path_string
    )

    if matches:
        return f"seed{matches[-1]}"

    return "unknown_seed"


# =============================================================================
# LOAD ONE RUN
# =============================================================================

def load_run_data(
    run_path: str,
    tag: str,
):

    path = Path(
        run_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Directory not found: {run_path}"
        )

    seed_name = extract_seed_name(
        run_path
    )

    reader = SummaryReader(
        path
    )

    df = reader.scalars

    df = df[
        df['tag'] == tag
    ].copy()

    if df.empty:

        raise ValueError(
            f"Tag '{tag}' not found in {run_path}"
        )

    df.rename(
        columns={
            'step': 'epoch'
        },
        inplace=True
    )

    df = df[
        [
            'epoch',
            'value'
        ]
    ].copy()

    df['epoch'] = pd.to_numeric(
        df['epoch'],
        errors='coerce'
    )

    df['value'] = pd.to_numeric(
        df['value'],
        errors='coerce'
    )

    df.dropna(
        subset=[
            'epoch',
            'value'
        ],
        inplace=True
    )

    df['epoch'] = (
        df['epoch']
        .astype(int)
    )

    # Keep the last recorded value if an epoch is duplicated.
    df = (
        df
        .sort_values('epoch')
        .drop_duplicates(
            subset='epoch',
            keep='last'
        )
    )

    epochs = df[
        'epoch'
    ].to_numpy()

    values = df[
        'value'
    ].to_numpy(
        dtype=float
    )

    return (
        seed_name,
        epochs,
        values
    )


# =============================================================================
# FIND COMMON EPOCHS ACROSS ALL OPTIMIZERS AND ALL SEEDS
# =============================================================================

def find_common_epochs(
    optimizers,
    tag,
):

    print()
    print("=" * 100)
    print(
        f"COMMON EPOCH INTERSECTION — {tag}"
    )
    print("=" * 100)

    all_epoch_sets = []

    for optimizer_name, run_paths in optimizers.items():

        print()
        print(
            f"[Optimizer] {optimizer_name}"
        )

        optimizer_epoch_sets = []

        for run_path in run_paths:

            (
                seed_name,
                epochs,
                values,
            ) = load_run_data(
                run_path,
                tag
            )

            epoch_set = set(
                epochs.tolist()
            )

            optimizer_epoch_sets.append(
                epoch_set
            )

            print(
                f"  {seed_name}: "
                f"{min(epoch_set)} -> "
                f"{max(epoch_set)} "
                f"({len(epoch_set)} epochs)"
            )


        if not optimizer_epoch_sets:
            print(
                f"  -> No runs found for {optimizer_name} / {tag}, skipping."
            )
            continue
        # Common epochs among all seeds
        # of this optimizer.
        optimizer_common = set.intersection(
            *optimizer_epoch_sets
        )

        if not optimizer_common:

            raise ValueError(
                f"No common epochs among "
                f"{optimizer_name} seeds."
            )

        print(
            f"  Optimizer common: "
            f"{min(optimizer_common)} -> "
            f"{max(optimizer_common)} "
            f"({len(optimizer_common)} epochs)"
        )

        all_epoch_sets.append(
            optimizer_common
        )

    # Common epochs across ALL optimizers.
    common_epochs = sorted(
        set.intersection(
            *all_epoch_sets
        )
    )

    if not common_epochs:

        raise ValueError(
            f"No common epochs across all optimizers "
            f"for {tag}."
        )

    print()
    print("-" * 100)
    print(
        "ALL OPTIMIZERS — COMMON EPOCHS"
    )
    print("-" * 100)

    print(
        f"Common range: "
        f"{common_epochs[0]} -> "
        f"{common_epochs[-1]}"
    )

    print(
        f"Number of common epochs: "
        f"{len(common_epochs)}"
    )

    if len(common_epochs) < LAST_N_EPOCHS:

        raise ValueError(
            f"Only {len(common_epochs)} common epochs "
            f"exist, but {LAST_N_EPOCHS} are required."
        )

    final_common_epochs = (
        common_epochs[
            -LAST_N_EPOCHS:
        ]
    )

    print()
    print(
        f"FINAL {LAST_N_EPOCHS} COMMON EPOCHS:"
    )

    print(
        f"{final_common_epochs[0]} -> "
        f"{final_common_epochs[-1]}"
    )

    print(
        f"Epoch list:"
    )

    print(
        final_common_epochs
    )

    print("=" * 100)

    return common_epochs


# =============================================================================
# EPOCH-WISE MEAN + STD
# =============================================================================

def load_optimizer_data(
    run_paths,
    tag,
    common_epochs,
):

    common_epoch_set = set(
        common_epochs
    )

    all_dfs = []

    for run_path in run_paths:

        (
            seed_name,
            epochs,
            values,
        ) = load_run_data(
            run_path,
            tag
        )

        filtered_rows = []

        for epoch, value in zip(
            epochs,
            values
        ):

            if epoch in common_epoch_set:

                filtered_rows.append(
                    {
                        'epoch': epoch,
                        'value': value,
                        'seed': seed_name
                    }
                )

        if not filtered_rows:

            continue

        all_dfs.append(
            pd.DataFrame(
                filtered_rows
            )
        )

    if not all_dfs:

        return (
            None,
            None,
            None,
            0
        )

    full = pd.concat(
        all_dfs,
        ignore_index=True
    )

    grouped = (
        full
        .groupby(
            'epoch'
        )['value']
        .apply(list)
    )

    # Keep exactly the common epoch ordering.
    grouped = grouped.reindex(
        common_epochs
    )

    epochs = grouped.index.to_numpy()

    means = []

    stds = []

    for vals in grouped:

        arr = np.asarray(
            vals,
            dtype=float
        )

        mean = np.mean(
            arr
        )

        if len(arr) > 1:

            std = np.std(
                arr,
                ddof=1
            )

        else:

            std = 0.0

        means.append(
            mean
        )

        stds.append(
            std
        )

    n_effective = len(
        all_dfs
    )

    return (
        epochs,
        np.asarray(means),
        np.asarray(stds),
        n_effective
    )


# =============================================================================
# EPOCH-WISE CSV
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
        exist_ok=True
    )

    rows = []

    for od in optimizers_data:

        for epoch, mean, std in zip(
            od['epochs'],
            od['mean'],
            od['std']
        ):

            rows.append({

                'epoch': epoch,

                'optimizer': od['label'],

                'mean': mean,

                'std': std,

                'std_lower': mean - std,

                'std_upper': mean + std,

                'n_seeds': od['n_seeds']

            })

    df = pd.DataFrame(
        rows
    )

    safe_tag = tag.replace(
        '/',
        '_'
    )

    fname = (
        out_path
        / f"{model_name}_{safe_tag}_common_summary.csv"
    )

    df.to_csv(
        fname,
        index=False,
        float_format="%.6f"
    )

    print(
        f"  💾 Saved common-range CSV to {fname}"
    )


# =============================================================================
# FINAL 30 COMMON EPOCHS — MEAN ± STD
# =============================================================================

def compute_final30_summary(
    optimizer_name,
    run_paths,
    tag,
    common_epochs,
):

    final_common_epochs = (
        common_epochs[
            -LAST_N_EPOCHS:
        ]
    )

    seed_means = {}

    for run_path in run_paths:

        (
            seed_name,
            epochs,
            values,
        ) = load_run_data(
            run_path,
            tag
        )

        epoch_to_value = dict(
            zip(
                epochs.tolist(),
                values.tolist()
            )
        )

        final_values = []

        for epoch in final_common_epochs:

            if epoch not in epoch_to_value:

                raise ValueError(
                    f"{optimizer_name} / "
                    f"{seed_name} / {tag}: "
                    f"missing common epoch {epoch}"
                )

            final_values.append(
                epoch_to_value[epoch]
            )

        final_values = np.asarray(
            final_values,
            dtype=float
        )

        seed_means[
            seed_name
        ] = float(
            np.mean(
                final_values
            )
        )

    seed_values = np.asarray(
        list(
            seed_means.values()
        ),
        dtype=float
    )

    n_seeds = len(
        seed_values
    )

    if n_seeds < 2:

        raise ValueError(
            f"{optimizer_name} / {tag}: "
            "at least two valid seeds are required."
        )

    mean_all_seeds = float(
        np.mean(
            seed_values
        )
    )

    std_all_seeds = float(
        np.std(
            seed_values,
            ddof=1
        )
    )

    return {

        'model': MODEL_NAME,

        'task': TASK,

        'dataset': DATASET,

        'metric': tag,

        'optimizer': optimizer_name,

        'n_seeds': n_seeds,

        'mean_all_seeds': mean_all_seeds,

        'std_all_seeds': std_all_seeds,

    }


# =============================================================================
# FINAL CSV — ONLY MEAN ± STD
# =============================================================================

def save_final_summary_csv(
    results,
    output_dir=OUTPUT_DIR,
):

    out_path = Path(
        output_dir
    )

    out_path.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for result in results:

        rows.append({

            'model': result['model'],

            'task': result['task'],

            'dataset': result['dataset'],

            'metric': result['metric'],

            'optimizer': result['optimizer'],

            'n_seeds': result['n_seeds'],

            'mean ± std': (
                f"{result['mean_all_seeds']:.4f}"
                f" ± "
                f"{result['std_all_seeds']:.4f}"
            ),

        })

    df = pd.DataFrame(
        rows
    )

    output_path = (
        out_path
        / (
            f"{MODEL_NAME}"
            f"_final_{LAST_N_EPOCHS}"
            f"_common_epochs_mean_std.csv"
        )
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 100)
    print(
        "FINAL COMMON-EPOCH SUMMARY"
    )
    print("=" * 100)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        f"💾 Saved final summary to:"
    )

    print(
        output_path
    )

    print("=" * 100)

    return df


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_final_results = []

    for tag in TAGS_TO_PLOT:

        print()
        print()
        print(
            "#" * 100
        )

        print(
            f"PROCESSING: {tag}"
        )

        print(
            "#" * 100
        )

        # =====================================================================
        # 1. FIND COMMON EPOCHS FIRST
        # =====================================================================

        common_epochs = find_common_epochs(
            OPTIMIZERS,
            tag
        )

        # =====================================================================
        # 2. PLOT ONLY COMMON EPOCHS
        # =====================================================================

        plt.figure(
            figsize=(12, 6)
        )

        any_plotted = False

        tag_data = []

        for idx, (
            label,
            dirs
        ) in enumerate(
            OPTIMIZERS.items()
        ):

            print(
                f"  {label} ..."
            )

            if not dirs:

                print(
                    "    -> No directories defined, skipping."
                )

                continue

            (
                epochs,
                mean,
                std,
                n_eff
            ) = load_optimizer_data(
                dirs,
                tag,
                common_epochs
            )

            if epochs is None:

                print(
                    "    -> No common data for this tag, skipping."
                )

                continue
            color = COLORS[label]

            # -----------------------------------------------------------------
            # Mean
            # -----------------------------------------------------------------

            plt.plot(
                epochs,
                mean,
                color=color,
                linewidth=2,
                label=label
            )

            # -----------------------------------------------------------------
            # ± STD
            # -----------------------------------------------------------------

            plt.fill_between(
                epochs,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
                label='_nolegend_'
            )

            any_plotted = True

            print(
                f"    Common range: "
                f"{epochs[0]} -> {epochs[-1]}"
            )

            print(
                f"    Effective seeds: {n_eff}"
            )

            tag_data.append({

                'label': label,

                'epochs': epochs,

                'mean': mean,

                'std': std,

                'n_seeds': n_eff

            })

            # ================================================================
            # 3. FINAL 30 COMMON EPOCHS
            # ================================================================

            try:

                final_result = (
                    compute_final30_summary(
                        label,
                        dirs,
                        tag,
                        common_epochs
                    )
                )

                all_final_results.append(
                    final_result
                )

                print(
                    f"    Final {LAST_N_EPOCHS} "
                    f"common epochs: "
                    f"{final_result['mean_all_seeds']:.4f} "
                    f"± "
                    f"{final_result['std_all_seeds']:.4f}"
                )

            except Exception as exc:

                print(
                    f"    ⚠️ Final summary failed: "
                    f"{exc}"
                )

        if not any_plotted:

            print(
                f"No common data for '{tag}'."
            )

            plt.close()

            continue

        # =====================================================================
        # 4. PLOT SETTINGS
        # =====================================================================

        plt.xlabel(
            'Epoch',
            fontsize=13
        )

        plt.ylabel(
            tag.replace(
                '_',
                ' '
            ).replace(
                'test',
                'validation'
            ),
            fontsize=13
        )


        plt.title(
            f'{tag.replace("test", "validation")} – '
            f'{MODEL_NAME} (4 seeds)',
            fontsize=16
        )

        plt.legend(
            fontsize=12
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        # =====================================================================
        # 5. SAVE PLOT
        # =====================================================================

        fname = (
            f'{MODEL_NAME}_'
            f'{tag.replace("/", "_")}_STD.svg'
        )

        plt.savefig(
            fname,
            dpi=200,
            bbox_inches='tight'
        )

        plt.show()

        print(
            f"  ✅ Saved {fname}"
        )

        # =====================================================================
        # 6. SAVE EPOCH-WISE COMMON-RANGE CSV
        # =====================================================================

        save_tag_csv(
            MODEL_NAME,
            tag,
            tag_data,
            OUTPUT_DIR
        )

    # =========================================================================
    # 7. SAVE FINAL 30-COMMON-EPOCH SUMMARY
    # =========================================================================

    if all_final_results:

        save_final_summary_csv(
            all_final_results,
            OUTPUT_DIR
        )

    else:

        print(
            "\nNo final common-epoch results were generated."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()