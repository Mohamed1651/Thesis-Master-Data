import os
import re
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats

PROJECT = "Shopizer"
VERSIONS = [
    "baseline",
    "execution-waste",
    "memory-friction",
    "boundary-overhead",
    "combined",
]
THRESHOLD = 262143.328850


def extract_metrics(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'PACKAGE_ENERGY (J)' not in df.columns or 'DRAM_ENERGY (J)' not in df.columns:
            print(f"Missing required columns in: {file_path}")
            return None
            
        pkg_diffs = df['PACKAGE_ENERGY (J)'].diff().mask(lambda x: x < 0, lambda x: x + THRESHOLD).sum()
        dram_diffs = df['DRAM_ENERGY (J)'].diff().mask(lambda x: x < 0, lambda x: x + THRESHOLD).sum()
        
        return {'CPU Energy (J)': pkg_diffs, 'DRAM Energy (J)': dram_diffs}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
def calculate_cliffs_delta(group1, group2):
    matrix = np.sign(np.subtract.outer(group1, group2))
    delta = np.mean(matrix)

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "Negligible"
    elif abs_delta < 0.330:
        magnitude = "Small"
    elif abs_delta < 0.474:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    return delta, magnitude


data_store = {}
for v in VERSIONS:
    v_path = os.path.join("../data", PROJECT, v)
    run_data = []
    if os.path.exists(v_path):
        for filename in [
            f for f in os.listdir(v_path) if re.match(r"^data\d+\.csv$", f)
        ]:
            metrics = extract_metrics(os.path.join(v_path, filename))
            if metrics is not None:
                metrics["version"] = v
                run_data.append(metrics)

    if run_data:
        data_store[v] = pd.DataFrame(run_data)

if not data_store:
    print("No valid data loaded. Check your folder structure/column names.")
else:
    full_df = pd.concat(data_store.values(), ignore_index=True)

    for metric in ["CPU Energy (J)", "DRAM Energy (J)"]:
        print(f"\n========================================================")
        print(f"METRIC: {metric}")
        print(f"========================================================")

        grouped_data = full_df.groupby("version")
        groups = [group[metric] for name, group in grouped_data]

        h_stat, p_val_global = stats.kruskal(*groups)
        print(f"Kruskal-Wallis H-statistic: {h_stat:.10f}")
        print(f"Global Kruskal-Wallis p-value: {p_val_global:.10f}")

        if p_val_global <= 0.05:
            print(
                "\nGlobal significance detected (p <= 0.05). Running Dunn's post-hoc test (Bonferroni adjusted)..."
            )

            # Generate pairwise matrix using scikit-posthocs
            dunn_matrix = sp.posthoc_dunn(
                full_df, val_col=metric, group_col="version", p_adjust="bonferroni"
            )

            baseline_key = "baseline"
            refactored_versions = [v for v in VERSIONS if v != baseline_key]

            print("\nPairwise Comparisons against Control Group:")
            print("-" * 75)
            print(f"{'Comparison Pair':<40} | {'Adjusted p-value':<18} | {'Status':<10} | {'Cliffs Delta':<15} | Magnitude")

            print("-" * 75)

            for version in refactored_versions:
                if (
                    version in dunn_matrix.index
                    and baseline_key in dunn_matrix.columns
                ):
                    p_adjusted = dunn_matrix.loc[version, baseline_key]
                    status = (
                        "SIGNIFICANT" if p_adjusted <= 0.05 else "NOT SIGNIFICANT"
                    )

                    baseline_data = full_df[full_df["version"] == baseline_key][
                        metric
                    ].values
                    refactored_data = full_df[full_df["version"] == version][
                        metric
                    ].values

                    # Run Cliff's Delta calculation
                    delta, magnitude = calculate_cliffs_delta(
                        refactored_data, baseline_data
                    )
                    print(
                        f"({baseline_key} vs. {version}) : | {p_adjusted:.10f} | {status} | Cliff's Delta: {delta:.4f} | ({magnitude})"
                    )
                else:
                    print(
                        f"({baseline_key} vs. {version}) : | Data Missing       | SKIPPED"
                    )
            print("-" * 75)
        else:
            print(
                "\nNo global significance detected (p > 0.05). Skipping post-hoc pairwise comparisons."
            )