import numpy as np

import re

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

from tbparse import SummaryReader

from pathlib import Path


# =============================================================================
# تنظیمات کاربر
# =============================================================================

MODEL_NAME = "ALBERT"

DATASET = "GLUE"

TASK = "mrpc"

# Number of final COMMON epochs used for the final summary.
LAST_N_EPOCHS = 8

# Output directory for CSV files.
OUTPUT_DIR = Path("csv_summaries")


# =============================================================================
# FIGURE / PREVIEW CONFIGURATION
# =============================================================================

FIGSIZE = (12, 6)

SAVE_DPI = 300

# Exact subplot layout
LEFT = 0.688
BOTTOM = 0.14
RIGHT = 0.996
TOP = 0.651
WSPACE = 0.2
HSPACE = 0.2

SAVE_FORMAT = "svg"

PAD_INCHES = 0.1


# =============================================================================
# OPTIMIZERS
# =============================================================================

OPTIMIZERS = {

    'AdaBelief': [

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadabelief_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_RECTFalse_SEED1402_20260621_222800',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadabelief_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_RECTFalse_SEED1993_20260621_223940',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadabelief_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_RECTFalse_SEED2025_20260621_225029',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadabelief_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_RECTFalse_SEED8_20260621_230120',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadabelief_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_RECTFalse_SEED11021979_20260622_113622',

    ],


    'Emoji (Proposed)': [

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTemoji_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_SEED1402_20260622_000933',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTemoji_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_SEED1993_20260622_002129',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTemoji_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_SEED2025_20260622_003325',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTemoji_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_SEED8_20260622_004405',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTemoji_B0.9-0.999_EPS1e-08_AMSGFalse_WDTrue_SEED11021979_20260622_005548',

    ],


    'AdaBound': [

        # مسیرهای مربوط به AdaBound را در اینجا قرار دهید (در صورت وجود)

    ],


    'AdamW': [

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadamw_B0.9-0.999_EPS1e-08_AMSGFalse_SEED1402_20260621_231240',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadamw_B0.9-0.999_EPS1e-08_AMSGFalse_SEED1993_20260621_232303',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadamw_B0.9-0.999_EPS1e-08_AMSGFalse_SEED2025_20260621_233405',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadamw_B0.9-0.999_EPS1e-08_AMSGFalse_SEED8_20260621_234558',

        'logs/mrpc_LR2e-05_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_WD0.01_OPTadamw_B0.9-0.999_EPS1e-08_AMSGFalse_SEED11021979_20260621_235811',

    ],


    'Yogi': [

        # مسیرهای مربوط به Yogi را در اینجا قرار دهید (در صورت وجود)

    ],


    'Lion': [

        'logs/mrpc_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_OPTlion_LR2e-06_B0.9-0.99_WD0.1_SEED1402_20260622_025301',

        'logs/mrpc_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_OPTlion_LR2e-06_B0.9-0.99_WD0.1_SEED1993_20260622_030237',

        'logs/mrpc_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_OPTlion_LR2e-06_B0.9-0.99_WD0.1_SEED2025_20260622_031203',

        'logs/mrpc_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_OPTlion_LR2e-06_B0.9-0.99_WD0.1_SEED8_20260622_032131',

        'logs/mrpc_BSZ8_GAS1_ADR0.0_CDR0.1_TS800_WS200_MSL512_OPTlion_LR2e-06_B0.9-0.99_WD0.1_SEED11021979_20260622_033057',

    ],


    'SGDW': [

        # مسیرهای مربوط به SGDW را در اینجا قرار دهید (در صورت وجود)

    ],

}


# =============================================================================
# METRICS
# =============================================================================

TAGS = [

    'train/loss',

    'eval/loss',

    'eval/accuracy',

    'eval/f1'

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

    '#7F7F7F'    # SGDW

]


# =============================================================================
# EXTRACT SEED NAME
# =============================================================================

def extract_seed_name(

    path_string: str,

) -> str:

    matches = re.findall(

        r"SEED(\d+)",

        path_string,

        flags=re.IGNORECASE

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

    active_optimizers = [

        optimizer_name

        for optimizer_name, run_paths

        in optimizers.items()

        if run_paths

    ]

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

            epoch_set = set(

                epochs.tolist()

            )

            if not epoch_set:

                raise ValueError(

                    f"No valid epochs in {run_path}"

                )

            seed_epoch_sets.append(

                epoch_set

            )

        optimizer_common = set.intersection(

            *seed_epoch_sets

        )

        if not optimizer_common:

            raise ValueError(

                f"No common epochs for "

                f"{optimizer_name} / {tag}"

            )

        optimizer_common_sets.append(

            optimizer_common

        )

    common_epochs = sorted(

        set.intersection(

            *optimizer_common_sets

        )

    )

    if not common_epochs:

        raise ValueError(

            f"No common epochs across all active "

            f"optimizers for {tag}"

        )

    return common_epochs


# =============================================================================
# EPOCH-WISE MEAN + SAMPLE STD
# =============================================================================

def load_optimizer_data(

    run_paths,

    tag,

):

    all_dfs = []

    for run_path in run_paths:

        path = Path(

            run_path

        )

        if not path.exists():

            print(

                f"  ⚠️ Directory not found: {run_path}"

            )

            continue

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

            print(

                f"  ⚠️ Tag '{tag}' not found in {run_path}"

            )

            continue

        df.rename(

            columns={

                'step': 'epoch'

            },

            inplace=True

        )

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

        df['seed'] = seed_name

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

        .groupby('epoch')['value']

        .apply(list)

    )

    steps = grouped.index.values

    means = []

    stds = []

    for vals in grouped:

        arr = np.asarray(

            vals,

            dtype=float

        )

        mean_value = np.mean(

            arr

        )

        if len(arr) > 1:

            std_value = np.std(

                arr,

                ddof=1

            )

        else:

            std_value = 0.0

        means.append(

            mean_value

        )

        stds.append(

            std_value

        )

    n_effective = len(

        all_dfs

    )

    return (

        steps,

        np.asarray(means),

        np.asarray(stds),

        n_effective

    )


# =============================================================================
# SAVE EPOCH-WISE CSV
# =============================================================================

def save_tag_csv(

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

        for step, mean, std in zip(

            od['steps'],

            od['mean'],

            od['std']

        ):

            rows.append({

                'step': step,

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

        /

        f"{MODEL_NAME}_{TASK}_"

        f"{safe_tag}_summary.csv"

    )

    df.to_csv(

        fname,

        index=False,

        float_format="%.6f"

    )

    print(

        f"  💾 Saved STD summary CSV to {fname}"

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

    if len(common_epochs) < LAST_N_EPOCHS:

        raise ValueError(

            f"{optimizer_name} / {tag}: "

            f"only {len(common_epochs)} common epochs "

            f"exist, but {LAST_N_EPOCHS} are required."

        )

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

        "model": MODEL_NAME,

        "task": TASK,

        "dataset": DATASET,

        "metric": tag,

        "optimizer": optimizer_name,

        "n_seeds": n_seeds,

        "final_epoch_start": (

            final_common_epochs[0]

        ),

        "final_epoch_end": (

            final_common_epochs[-1]

        ),

        "mean_all_seeds": (

            mean_all_seeds

        ),

        "std_all_seeds": (

            std_all_seeds

        ),

        "mean ± std": (

            f"{mean_all_seeds:.4f}"

            f" ± "

            f"{std_all_seeds:.4f}"

        ),

    }


# =============================================================================
# SAVE FINAL SUMMARY CSV
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

    df = pd.DataFrame(

        results

    )

    output_path = (

        out_path

        /

        f"{MODEL_NAME}_{TASK}_final_"

        f"{LAST_N_EPOCHS}_common_epochs_mean_std.csv"

    )

    df.to_csv(

        output_path,

        index=False,

        float_format="%.6f"

    )

    print()

    print(

        "=" * 100

    )

    print(

        "FINAL 30 COMMON-EPOCH SUMMARY"

    )

    print(

        "=" * 100

    )

    print(

        df.to_string(

            index=False

        )

    )

    print()

    print(

        "💾 Saved final summary to:"

    )

    print(

        output_path

    )

    print(

        "=" * 100

    )


# =============================================================================
# PLOT + AUTO-SAVE ON CLOSE
# =============================================================================

def plot_tag(

    tag,

    tag_data,

    output_path,

):

    fig, ax = plt.subplots(

        figsize=FIGSIZE

    )

    # Exact preview layout
    fig.subplots_adjust(

        left=LEFT,

        bottom=BOTTOM,

        right=RIGHT,

        top=TOP,

        wspace=WSPACE,

        hspace=HSPACE

    )

    for od in tag_data:

        label = od["label"]

        steps = od["steps"]

        mean = od["mean"]

        std = od["std"]

        color = od["color"]

        plot_kwargs = {

            "color": color,

            "linewidth": 2,

            "label": label

        }

        ax.plot(

            steps,

            mean,

            **plot_kwargs

        )

        # Mean ± Sample STD
        ax.fill_between(

            steps,

            mean - std,

            mean + std,

            color=color,

            alpha=0.15,

            label="_nolegend_"

        )


    ax.set_xlabel(

        "Step",

        fontsize=13

    )


    ax.set_ylabel(

        tag.replace(

            "_",

            " "

        ),

        fontsize=13

    )


    ax.set_title(

        f"{tag} – "

        f"{TASK.upper()} on "

        f"{DATASET} using "

        f"{MODEL_NAME}",

        fontsize=16

    )


    ax.legend(

        fontsize=12

    )


    ax.grid(

        alpha=0.3

    )


    saved = {

        "done": False

    }


    def save_on_close(event):

        if saved["done"]:

            return

        fig.savefig(

            output_path,

            format=SAVE_FORMAT,

            dpi=SAVE_DPI,

            bbox_inches="tight",

            pad_inches=PAD_INCHES

        )

        saved["done"] = True

        print()

        print(

            "✅ Preview closed."

        )

        print(

            f"💾 SVG saved to: {output_path}"

        )


    fig.canvas.mpl_connect(

        "close_event",

        save_on_close

    )


    print()

    print(

        f"Preview: {tag}"

    )

    print(

        "Close the preview window to save the SVG."

    )

    print()


    plt.show(

        block=True

    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    all_final_results = []


    for tag in TAGS:

        print()

        print(

            f"--- Tag: {tag} ---"

        )


        # =====================================================================
        # 1. FIND COMMON EPOCHS
        # =====================================================================

        try:

            common_epochs = find_common_epochs(

                OPTIMIZERS,

                tag

            )

            print(

                f"Common epoch range: "

                f"{common_epochs[0]} -> "

                f"{common_epochs[-1]}"

            )

            if len(common_epochs) < LAST_N_EPOCHS:

                print(

                    f"⚠️ Only {len(common_epochs)} "

                    f"common epochs available."

                )

            else:

                final_epochs = (

                    common_epochs[

                        -LAST_N_EPOCHS:

                    ]

                )

                print(

                    f"Final {LAST_N_EPOCHS} "

                    f"common epochs: "

                    f"{final_epochs[0]} -> "

                    f"{final_epochs[-1]}"

                )

        except Exception as exc:

            print(

                f"⚠️ Common-epoch analysis failed "

                f"for {tag}: {exc}"

            )

            continue


        # =====================================================================
        # 2. LOAD OPTIMIZER DATA
        # =====================================================================

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

                    "    -> No active directories, skipping."

                )

                continue


            (

                steps,

                mean,

                std,

                n_eff

            ) = load_optimizer_data(

                dirs,

                tag

            )


            if steps is None:

                print(

                    "    -> No data for this tag, skipping."

                )

                continue


            color = COLORS[

                idx % len(COLORS)

            ]


            if len(mean) > 0:

                print(

                    f"    Effective seeds: "

                    f"{n_eff}, "

                    f"Final mean={mean[-1]:.4f}, "

                    f"STD ±{std[-1]:.4f}"

                )


            tag_data.append({

                "label": label,

                "steps": steps,

                "mean": mean,

                "std": std,

                "n_seeds": n_eff,

                "color": color

            })


            # =================================================================
            # 3. FINAL 30 COMMON EPOCHS
            # =================================================================

            if common_epochs is not None:

                try:

                    final_result = (

                        compute_final30_summary(

                            optimizer_name=label,

                            run_paths=dirs,

                            tag=tag,

                            common_epochs=common_epochs

                        )

                    )

                    all_final_results.append(

                        final_result

                    )

                    print(

                        f"    Final "

                        f"{LAST_N_EPOCHS} "

                        f"common epochs: "

                        f"{final_result['mean_all_seeds']:.4f}"

                        f" ± "

                        f"{final_result['std_all_seeds']:.4f}"

                    )

                except Exception as exc:

                    print(

                        f"    ⚠️ Final summary failed: "

                        f"{exc}"

                    )


        if not tag_data:

            print(

                f"  No data for tag '{tag}'."

            )

            continue


        # =====================================================================
        # 4. OUTPUT PATH
        # =====================================================================

        safe_tag = tag.replace(

            "/",

            "_"

        )

        output_path = (

            Path(

                f"{MODEL_NAME}_"

                f"{TASK}_"

                f"{safe_tag}_STD.svg"

            )

        )


        # =====================================================================
        # 5. PREVIEW + SAVE ON CLOSE
        # =====================================================================

        plot_tag(

            tag=tag,

            tag_data=tag_data,

            output_path=output_path

        )


        # =====================================================================
        # 6. EPOCH-WISE CSV
        # =====================================================================

        save_tag_csv(

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

            "\n⚠️ No final 30-epoch summaries were generated."

        )


    print()

    print(

        "✅ All MRPC plots and CSV files generated."

    )

    print(

        "Check the 'csv_summaries' folder."

    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()