import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


# =============================================================================
# USER CONFIGURATION
# =============================================================================

MODEL_NAME = "ResNet34"

# Number of final epochs used for the final per-seed summary.
LAST_N_EPOCHS = 30

# Output directory for CSV files.
OUTPUT_DIR = Path("csv_summaries")


# =============================================================================
# OPTIMIZERS
# =============================================================================

OPTIMIZERS = {

    'Original AdaBelief': [

        'runs/resnet34-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed0-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/resnet34-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1402-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/resnet34-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed2025-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-20251012-205623-resetFalse',

        'runs/resnet34-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed8-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse'

    ],


    'α=0 (No centering)': [

        'runs/resnet34-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed1402-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260521-062712-resetFalse',

        'runs/resnet34-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed2025-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260521-091452-resetFalse',

        'runs/resnet34-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed8-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260521-150130-resetFalse',

        'runs/resnet34-emojiabl-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed0-alphaa0.0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1--wd0.01-20260524-075002-resetFalse',

    ],



    'α=1 (Emoji)': [

        'runs/resnet34-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed0-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/resnet34-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed1402-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/resnet34-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed2025-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

        'runs/resnet34-emoji-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.01-run0-seed8-wdcpl-warmupsteplr--linT750-linS0.001-slrSS140-slrGam0.1-resetFalse',

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
) -> tuple[
    str,
    np.ndarray,
    np.ndarray,
]:

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
# EXISTING EPOCH-WISE MEAN + STD
# =============================================================================

def load_optimizer_data(
    seed_dirs,
    tag,
):

    all_dfs = []

    for i, d in enumerate(
        seed_dirs
    ):

        p = Path(
            d
        )

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
# FINAL 30-EPOCH SUMMARY
# =============================================================================

def compute_final30_summary(
    optimizer_name: str,
    run_paths: list[str],
    tag: str,
) -> dict:

    """
    For each seed:
        1. Take the final LAST_N_EPOCHS.
        2. Compute the mean of those epochs.

    Then across seed-level means:
        3. Compute the overall mean.
        4. Compute the sample STD (ddof=1).
    """

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

        if len(values) < LAST_N_EPOCHS:

            raise ValueError(
                f"{optimizer_name} / {tag} / {seed_name}: "
                f"only {len(values)} epochs available; "
                f"{LAST_N_EPOCHS} required."
            )

        final_epochs = (
            epochs[
                -LAST_N_EPOCHS:
            ]
        )

        final_values = (
            values[
                -LAST_N_EPOCHS:
            ]
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
# FINAL 30-EPOCH CSV — ONLY MEAN ± STD
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
            f"_epochs_mean_std.csv"
        )
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()

    print(
        "=" * 80
    )

    print(
        f"💾 Saved final "
        f"{LAST_N_EPOCHS}-epoch summary:"
    )

    print(
        output_path
    )

    print(
        "=" * 80
    )

    print()

    print(
        df.to_string(
            index=False
        )
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Results for final 30-epoch summary.
    all_final30_results = []

    for tag in TAGS_TO_PLOT:

        print()

        print(
            f"{'=' * 60}"
        )

        print(
            f"Processing tag: {tag}"
        )

        print(
            f"{'=' * 60}"
        )

        # ---------------------------------------------------------------------
        # EXISTING PLOT
        # ---------------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # Existing epoch-wise Mean ± STD
            # -----------------------------------------------------------------

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

            color = COLORS[
                idx % len(COLORS)
            ]

            plt.plot(
                epochs,
                mean,
                color=color,
                linewidth=2,
                label=label
            )

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
                f"    Runs (effective): {n_eff}, "
                f"Final: mean={mean[-1]:.4f}, "
                f"STD ±{std[-1]:.4f}"
            )

            tag_data.append({

                'label': label,

                'epochs': epochs,

                'mean': mean,

                'std': std,

                'n_seeds': n_eff

            })

            # -----------------------------------------------------------------
            # NEW:
            # Final 30 epochs:
            # per-seed mean -> mean + STD across seeds
            # -----------------------------------------------------------------

            try:

                final30_result = (
                    compute_final30_summary(
                        label,
                        dirs,
                        tag
                    )
                )

                all_final30_results.append(
                    final30_result
                )

                print(
                    f"    Final {LAST_N_EPOCHS} epochs: "
                    f"mean ± std = "
                    f"{final30_result['mean_all_seeds']:.4f} "
                    f"± "
                    f"{final30_result['std_all_seeds']:.4f}"
                )

            except Exception as exc:

                print(
                    f"    ⚠️ Final-{LAST_N_EPOCHS}-epoch "
                    f"summary failed: {exc}"
                )

        # ---------------------------------------------------------------------
        # Finish plot
        # ---------------------------------------------------------------------

        if not any_plotted:

            print(
                f"  No data for tag '{tag}' "
                "in any optimiser. Skipping figure."
            )

            plt.close()

            continue

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

        fname = (
            f'resnet34_'
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

        # ---------------------------------------------------------------------
        # Existing epoch-wise CSV
        # ---------------------------------------------------------------------

        save_tag_csv(
            MODEL_NAME,
            tag,
            tag_data,
            OUTPUT_DIR
        )

    # =========================================================================
    # NEW FINAL 30-EPOCH CSV
    # =========================================================================

    if all_final30_results:

        save_final30_summary_csv(
            all_final30_results,
            OUTPUT_DIR
        )

    else:

        print(
            "\nNo final-30-epoch results were generated."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()