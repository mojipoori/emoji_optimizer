import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


# =============================================================================
# USER CONFIGURATION – edit this section only
# =============================================================================

MODEL_NAME = "ResNet20"

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

        # 'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-082007-resetFalse',

        # 'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-104138-resetFalse',

        # 'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-055840-resetFalse',

        # 'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-192257-resetFalse'


        'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20260529-141148-resetFalse',
        'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20260529-170743-resetFalse',
        'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20260529-201633-resetFalse',
        'runs/resnet20-adabelief-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-rectifyFalse-fixed_decayFalse-amsgradFalse-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20260530-004053-resetFalse'

    ],

    'Emoji (Proposed)': [

      #'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed8-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-050112-resetFalse',

      #'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed2025-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-225509-resetFalse',

      #'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed1402-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-033242-resetFalse',

      #'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed1993-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-010744-resetFalse',


        'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed1993-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1--wd0.0005-20260529-153653-resetFalse',
        'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed8-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1--wd0.0005-20260529-183924-resetFalse',
        'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed1402-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1--wd0.0005-20260529-214627-resetFalse',
        'runs/resnet20-emoji-lr0.01-betas0.9-0.99-eps1e-08-wdecay0.0005-run0-seed2025-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1--wd0.0005-20260530-022648-resetFalse'

    ],

    # ---- NEW OPTIMISERS ------------------------------------------------------

    'AdaBound': [

        'runs/resnet20-adaboundw-lr0.001-betas0.9-0.99-final_lr0.1-gamma0.001-wdecay0.0005-eps1e-08-amsboundFalse-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-212619-resetFalse',

        'runs/resnet20-adaboundw-lr0.001-betas0.9-0.99-final_lr0.1-gamma0.001-wdecay0.0005-eps1e-08-amsboundFalse-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-010156-resetFalse',

        'runs/resnet20-adaboundw-lr0.001-betas0.9-0.99-final_lr0.1-gamma0.001-wdecay0.0005-eps1e-08-amsboundFalse-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-044037-resetFalse',

        'runs/resnet20-adaboundw-lr0.001-betas0.9-0.99-final_lr0.1-gamma0.001-wdecay0.0005-eps1e-08-amsboundFalse-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-081817-resetFalse',

    ],

    'AdamW': [

        'runs/resnet20-adamw-lr0.01-betas0.9-0.99-wdecay0.0005-eps1e-08-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-203642-resetFalse',

        'runs/resnet20-adamw-lr0.01-betas0.9-0.99-wdecay0.0005-eps1e-08-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-071146-resetFalse',

        'runs/resnet20-adamw-lr0.01-betas0.9-0.99-wdecay0.0005-eps1e-08-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-093314-resetFalse',

        'runs/resnet20-adamw-lr0.01-betas0.9-0.99-wdecay0.0005-eps1e-08-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-115436-resetFalse',

    ],

    'Yogi': [

        'runs/resnet20-yogi-lr0.01-betas0.9-0.99-eps0.001-wdecay0.0005-run0-seed8-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-201326-resetFalse',

        'runs/resnet20-yogi-lr0.01-betas0.9-0.99-eps0.001-wdecay0.0005-run0-seed2025-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-234927-resetFalse',

        'runs/resnet20-yogi-lr0.01-betas0.9-0.99-eps0.001-wdecay0.0005-run0-seed1993-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-032742-resetFalse',

        'runs/resnet20-yogi-lr0.01-betas0.9-0.99-eps0.001-wdecay0.0005-run0-seed1402-wdcpl-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-070525-resetFalse',

    ],

    'Lion': [

        'runs/resnet20-lion-lr0.001-betas0.9-0.99-wdecay0.005-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-224003-resetFalse',

        'runs/resnet20-lion-lr0.001-betas0.9-0.99-wdecay0.005-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-021546-resetFalse',

        'runs/resnet20-lion-lr0.001-betas0.9-0.99-wdecay0.005-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-055359-resetFalse',

        'runs/resnet20-lion-lr0.001-betas0.9-0.99-wdecay0.005-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251008-093135-resetFalse',

    ],

    'SGDW': [

        'runs/resnet20-sgdw-lr0.1-mom0.9-wd0.0005-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-145709-resetFalse',

        'runs/resnet20-sgdw-lr0.1-mom0.9-wd0.0005-run0-seed2025-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251006-214452-resetFalse',

        'runs/resnet20-sgdw-lr0.1-mom0.9-wd0.0005-run0-seed1993-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251007-022215-resetFalse',

        'runs/resnet20-sgdw-lr0.1-mom0.9-wd0.0005-run0-seed1402-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20260509-113315-resetFalse',

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

    '#D62728',   # AdaBelief
    '#1F77B4',   # Emoji (Proposed)
    '#2CA02C',   # AdaBound
    '#9467BD',   # AdamW
    '#8C564B',   # Yogi
    '#E377C2',   # Lion
    '#7F7F7F',   # SGDW

]


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
# FIND COMMON EPOCHS
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

    active_optimizers = []

    for optimizer_name, run_paths in optimizers.items():

        # Completely commented-out optimizer.
        if not run_paths:

            print()
            print(
                f"[Optimizer] {optimizer_name}: "
                f"SKIPPED — no active run paths."
            )

            continue

        active_optimizers.append(
            optimizer_name
        )

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

        # Common epochs across all seeds
        # of this optimizer.
        optimizer_common = set.intersection(
            *optimizer_epoch_sets
        )

        print(
            f"  Optimizer common: "
            f"{min(optimizer_common)} -> "
            f"{max(optimizer_common)} "
            f"({len(optimizer_common)} epochs)"
        )

    if not active_optimizers:

        raise ValueError(
            "No active optimizers have run paths."
        )

    optimizer_common_sets = []

    for optimizer_name in active_optimizers:

        run_paths = optimizers[
            optimizer_name
        ]

        seed_epoch_sets = []

        for run_path in run_paths:

            (
                seed_name,
                epochs,
                values,
            ) = load_run_data(
                run_path,
                tag
            )

            seed_epoch_sets.append(
                set(
                    epochs.tolist()
                )
            )

        optimizer_common_sets.append(
            set.intersection(
                *seed_epoch_sets
            )
        )

    # Common epochs across ALL active optimizers.
    common_epochs = sorted(
        set.intersection(
            *optimizer_common_sets
        )
    )

    if not common_epochs:

        raise ValueError(
            f"No common epochs across active optimizers "
            f"for {tag}."
        )

    print()
    print("-" * 100)
    print(
        "ALL ACTIVE OPTIMIZERS — COMMON EPOCHS"
    )
    print("-" * 100)

    print(
        f"Active optimizers: "
        f"{active_optimizers}"
    )

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
        "Epoch list:"
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

        rows = []

        for epoch, value in zip(
            epochs,
            values
        ):

            if epoch in common_epoch_set:

                rows.append({

                    'epoch': epoch,

                    'value': value,

                    'seed': seed_name

                })

        if rows:

            all_dfs.append(
                pd.DataFrame(
                    rows
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

        means.append(mean)
        stds.append(std)

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

        seed_means[
            seed_name
        ] = float(
            np.mean(
                np.asarray(
                    final_values,
                    dtype=float
                )
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

            # Skip optimizer with all paths commented out.
            if not dirs:

                print(
                    "    -> No active directories, skipping."
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

            color = COLORS[
                idx % len(COLORS)
            ]

            # -----------------------------------------------------------------
            # MEAN
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
                f"    Effective seeds: "
                f"{n_eff}"
            )

            print(
                f"    Final common epoch: "
                f"mean={mean[-1]:.4f}, "
                f"STD ±{std[-1]:.4f}"
            )

            tag_data.append({

                'label': label,

                'epochs': epochs,

                'mean': mean,

                'std': std,

                'n_seeds': n_eff

            })

            # =================================================================
            # 3. FINAL 30 COMMON EPOCHS
            # =================================================================

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
        # 6. EPOCH-WISE CSV
        # =====================================================================

        save_tag_csv(
            MODEL_NAME,
            tag,
            tag_data,
            OUTPUT_DIR
        )

    # =========================================================================
    # 7. FINAL SUMMARY CSV
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