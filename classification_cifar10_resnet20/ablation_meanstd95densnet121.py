import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


# =============================================================================
# USER CONFIGURATION
# =============================================================================

MODEL_NAME = "DenseNet121"

# Number of final common epochs used for the final summary.
LAST_N_EPOCHS = 30

# Output directory for CSV files.
OUTPUT_DIR = Path("csv_summaries")


# =============================================================================
# OPTIMIZERS
# =============================================================================

OPTIMIZERS = {

    'Original AdaBelief': [

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-055632-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-062442-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-093547-resetFalse',

        'runs/densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-151245-resetFalse',

    ],

    'α=0 (No centering)': [

        'runs/densenet-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed2025-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260522-100325-resetFalse',

        'runs/densenet-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed0-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260522-135554-resetFalse',

        'runs/densenet-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed1402-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260523-191451-resetFalse',

        'runs/densenet-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed8-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260523-230637-resetFalse',

    ],



    'α=1 (Emoji)': [

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed8-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-091625-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed2025-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-130029-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251019-163922-resetFalse',

        'runs/densenet-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed1402-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251020-003415-resetFalse',

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

COLORS = [
    '#D62728',
    '#FF7F0E',
    '#1F77B4',
    '#2CA02C'
]


# =============================================================================
# HELPER: EXTRACT SEED NAME
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
# LOAD ONE SEED
# =============================================================================

def load_seed_data(
    run_path: str,
    tag: str,
):

    path = Path(run_path)

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
    optimizer_dict,
    tag,
):

    print()
    print("=" * 90)
    print(
        f"COMMON EPOCH INTERSECTION — {tag}"
    )
    print("=" * 90)

    optimizer_common_sets = []

    for optimizer_name, run_paths in optimizer_dict.items():

        print()
        print(
            f"[Optimizer] {optimizer_name}"
        )

        seed_epoch_sets = []

        for run_path in run_paths:

            (
                seed_name,
                epochs,
                values,
            ) = load_seed_data(
                run_path,
                tag
            )

            epoch_set = set(
                epochs.tolist()
            )

            seed_epoch_sets.append(
                epoch_set
            )

            print(
                f"  {seed_name}: "
                f"{min(epoch_set)} -> "
                f"{max(epoch_set)} "
                f"({len(epoch_set)} epochs)"
            )

        if not seed_epoch_sets:

            raise ValueError(
                f"No valid seed data for "
                f"{optimizer_name} / {tag}"
            )

        # Epochs shared by ALL seeds of this optimizer.
        optimizer_common = set.intersection(
            *seed_epoch_sets
        )

        if not optimizer_common:

            raise ValueError(
                f"No common epochs among seeds of "
                f"{optimizer_name} / {tag}"
            )

        print(
            f"  Optimizer common range: "
            f"{min(optimizer_common)} -> "
            f"{max(optimizer_common)} "
            f"({len(optimizer_common)} epochs)"
        )

        optimizer_common_sets.append(
            optimizer_common
        )

    # Epochs shared by ALL optimizers.
    common_epochs = sorted(
        set.intersection(
            *optimizer_common_sets
        )
    )

    if not common_epochs:

        raise ValueError(
            f"No common epochs across all optimizers "
            f"for metric '{tag}'."
        )

    print()
    print("-" * 90)
    print(
        "ALL OPTIMIZERS — COMMON EPOCHS"
    )
    print("-" * 90)

    print(
        f"Common epoch range: "
        f"{common_epochs[0]} -> "
        f"{common_epochs[-1]}"
    )

    print(
        f"Number of common epochs: "
        f"{len(common_epochs)}"
    )

    if len(common_epochs) >= LAST_N_EPOCHS:

        final_common_epochs = (
            common_epochs[
                -LAST_N_EPOCHS:
            ]
        )

        print()
        print(
            f"Final {LAST_N_EPOCHS} common epochs:"
        )

        print(
            f"{final_common_epochs[0]} "
            f"-> "
            f"{final_common_epochs[-1]}"
        )

        print(
            f"Epoch list: "
            f"{final_common_epochs}"
        )

    else:

        raise ValueError(
            f"Only {len(common_epochs)} common epochs exist, "
            f"but {LAST_N_EPOCHS} are required."
        )

    print("=" * 90)

    return common_epochs


# =============================================================================
# EXISTING EPOCH-WISE MEAN + STD
# =============================================================================

def load_optimizer_data(
    seed_dirs,
    tag,
):

    all_dfs = []

    for d in seed_dirs:

        p = Path(d)

        if not p.exists():

            raise FileNotFoundError(
                f"Directory not found: {d}"
            )

        seed_name = extract_seed_name(
            d
        )

        reader = SummaryReader(
            p
        )

        df = reader.scalars

        df = df[
            df['tag'] == tag
        ].copy()

        if df.empty:

            print(
                f"  ⚠️ Tag '{tag}' not found in {d}"
            )

            continue

        df['seed'] = seed_name

        df.rename(
            columns={
                'step': 'epoch'
            },
            inplace=True
        )

        all_dfs.append(
            df[
                [
                    'epoch',
                    'value',
                    'seed'
                ]
            ]
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

    epochs = grouped.index.values

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
# EXISTING EPOCH-WISE CSV
# =============================================================================

def save_tag_csv(
    model_name,
    tag,
    optimizers_data,
    output_dir="csv_summaries"
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
        / f"{model_name}_{safe_tag}_summary.csv"
    )

    df.to_csv(
        fname,
        index=False,
        float_format="%.6f"
    )

    print(
        f"  💾 Saved epoch-wise CSV to {fname}"
    )


# =============================================================================
# FINAL 30 COMMON EPOCH SUMMARY
# =============================================================================

def compute_final30_summary(
    optimizer_name: str,
    run_paths: list[str],
    tag: str,
    common_epochs,
) -> dict:

    """
    1. Restrict every seed to the common epoch range.
    2. Take the final LAST_N_EPOCHS from that common range.
    3. Average those epochs separately for each seed.
    4. Compute mean ± sample STD across the seed-level means.
    """

    final_common_epochs = common_epochs[
        -LAST_N_EPOCHS:
    ]

    seed_means = {}

    for run_path in run_paths:

        (
            seed_name,
            epochs,
            values,
        ) = load_seed_data(
            run_path,
            tag
        )

        epoch_to_value = dict(
            zip(
                epochs.tolist(),
                values.tolist()
            )
        )

        try:

            final_values = np.asarray(
                [
                    epoch_to_value[epoch]
                    for epoch in final_common_epochs
                ],
                dtype=float
            )

        except KeyError as exc:

            raise ValueError(
                f"{optimizer_name} / {tag} / {seed_name}: "
                f"missing common epoch {exc}"
            )

        if len(final_values) != LAST_N_EPOCHS:

            raise ValueError(
                f"{optimizer_name} / {tag} / {seed_name}: "
                f"expected {LAST_N_EPOCHS} common epochs, "
                f"found {len(final_values)}"
            )

        seed_mean = float(
            np.mean(
                final_values
            )
        )

        seed_means[
            seed_name
        ] = seed_mean

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

        'metric': tag,

        'optimizer': optimizer_name,

        'n_seeds': n_seeds,

        'mean_all_seeds': mean_all_seeds,

        'std_all_seeds': std_all_seeds,

    }


# =============================================================================
# FINAL CSV — ONLY MEAN ± STD
# =============================================================================

def save_final30_summary_csv(
    all_results,
    output_dir=OUTPUT_DIR,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for result in all_results:

        rows.append({

            'model': result['model'],

            'metric': result['metric'],

            'optimizer': result['optimizer'],

            'n_seeds': result['n_seeds'],

            'mean ± std': (
                f"{result['mean_all_seeds']:.4f} "
                f"± "
                f"{result['std_all_seeds']:.4f}"
            ),

        })

    df = pd.DataFrame(
        rows
    )

    output_path = (
        output_dir
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
    print("=" * 90)
    print(
        "FINAL COMMON-EPOCH SUMMARY"
    )
    print("=" * 90)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to:\n{output_path}"
    )

    print("=" * 90)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_final30_results = []

    for tag in TAGS_TO_PLOT:

        # =====================================================================
        # 1. FIND COMMON EPOCH RANGE FIRST
        # =====================================================================

        common_epochs = find_common_epochs(
            OPTIMIZERS,
            tag
        )

        # =====================================================================
        # 2. PLOT — RESTRICTED TO COMMON EPOCH RANGE
        # =====================================================================

        plt.figure(
            figsize=(12, 6)
        )

        any_plotted = False

        tag_data = []

        common_epoch_set = set(
            common_epochs
        )

        for idx, (
            label,
            dirs
        ) in enumerate(
            OPTIMIZERS.items()
        ):

            print(
                f"  Processing {label} ..."
            )

            (
                epochs,
                mean,
                std,
                n_eff
            ) = load_optimizer_data(
                dirs,
                tag
            )

            if epochs is None:

                print(
                    "    -> No data for this tag, skipping."
                )

                continue

            # Keep only common epochs.
            mask = np.asarray(
                [
                    epoch in common_epoch_set
                    for epoch in epochs
                ],
                dtype=bool
            )

            plot_epochs = epochs[
                mask
            ]

            plot_mean = mean[
                mask
            ]

            plot_std = std[
                mask
            ]

            if len(plot_epochs) == 0:

                print(
                    "    -> No common epochs, skipping."
                )

                continue

            color = COLORS[
                idx % len(COLORS)
            ]

            plt.plot(
                plot_epochs,
                plot_mean,
                color=color,
                linewidth=2,
                label=label
            )

            plt.fill_between(
                plot_epochs,
                plot_mean - plot_std,
                plot_mean + plot_std,
                color=color,
                alpha=0.15,
                label='_nolegend_'
            )

            any_plotted = True

            print(
                f"    Common range used: "
                f"{plot_epochs[0]} -> "
                f"{plot_epochs[-1]}"
            )

            tag_data.append({

                'label': label,

                'epochs': plot_epochs,

                'mean': plot_mean,

                'std': plot_std,

                'n_seeds': n_eff

            })

            # ================================================================
            # 3. FINAL 30 COMMON EPOCHS
            # ================================================================

            try:

                final30_result = (
                    compute_final30_summary(
                        label,
                        dirs,
                        tag,
                        common_epochs
                    )
                )

                all_final30_results.append(
                    final30_result
                )

                print(
                    f"    Final {LAST_N_EPOCHS} "
                    f"common epochs: "
                    f"{final30_result['mean_all_seeds']:.4f} "
                    f"± "
                    f"{final30_result['std_all_seeds']:.4f}"
                )

            except Exception as exc:

                print(
                    f"    ⚠️ Final-{LAST_N_EPOCHS}-epoch "
                    f"summary failed: {exc}"
                )

        if not any_plotted:

            print(
                f"  No common data for '{tag}'."
            )

            plt.close()

            continue

        # =====================================================================
        # PLOT SETTINGS
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
        # SAVE PLOT
        # =====================================================================

        fname = (
            f'densenet121_'
            f'{tag.replace("/", "_")}_STD.png'
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
        # SAVE EPOCH-WISE CSV — COMMON RANGE ONLY
        # =====================================================================

        save_tag_csv(
            MODEL_NAME,
            tag,
            tag_data,
            OUTPUT_DIR
        )

    # =========================================================================
    # FINAL 30 COMMON-EPOCH CSV
    # =========================================================================

    if all_final30_results:

        save_final30_summary_csv(
            all_final30_results,
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