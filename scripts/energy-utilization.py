import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np

def extract_metrics(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    threshold = 262143.328850
    try:
        # Calculate energy by summing positive differences between samples
        # This handles counter wraps or resets automatically
        pkg_diffs = df['PACKAGE_ENERGY (J)'].diff()
        pkg_diffs = pkg_diffs.mask(pkg_diffs < 0, pkg_diffs + threshold)
        pkg_energy = pkg_diffs.sum()
        
        dram_diffs = df['DRAM_ENERGY (J)'].diff()
        dram_diffs = dram_diffs.mask(dram_diffs < 0, dram_diffs + threshold)
        dram_energy = dram_diffs.sum()
        
        return {
            'CPU Energy (J)': pkg_energy,
            'DRAM Energy (J)': dram_energy,
            'filename': os.path.basename(file_path)
        }
    except KeyError as e:
        print(f"Missing column in {file_path}: {e}")
        return None


base_path = 'data/Train-Ticket/'
output_path = 'output/Train-Ticket/energy-usage/'
versions = ['baseline', 'execution-waste', 'memory-friction', 'combined']
plt.rcParams.update({
            'figure.dpi': 300,
            'font.size': 26,
            'axes.titlesize': 26,
            'axes.labelsize': 26,
            'xtick.labelsize': 26,
            'ytick.labelsize': 26,
            'legend.fontsize': 26
})

for v in versions:
    v_path = os.path.join(base_path, v)
    
    run_data = []
    files = [f for f in os.listdir(v_path) if re.match(r'^data\d+\.csv$', f)]
    
    for filename in files:
        stats = extract_metrics(os.path.join(v_path, filename))
        if stats:
            run_data.append(stats)
            
    if run_data:
        summary_df = pd.DataFrame(run_data)
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        fig.suptitle(f'Train-Ticket - {v.replace("-", " ").title()} Energy Usage', fontsize=30, fontweight='bold', y=0.98)
        metrics = [('CPU Energy (J)', 'green', 'Energy (Package)'), 
                   ('DRAM Energy (J)', 'blue', 'Energy (DRAM)')]

        for i, (col, color, title) in enumerate(metrics):
            sns.boxplot(y=summary_df[col], ax=axes[i], color='white', showfliers=False, linewidth=3, width=0.5)
            sns.stripplot(y=summary_df[col], ax=axes[i], color=color, jitter=True, size=8)
            
            mean_val = summary_df[col].mean()
            std_val = summary_df[col].std()
            axes[i].axhline(mean_val, color='red', linestyle='--', linewidth=3, label=f'Mean: {mean_val:.2f}')
            axes[i].axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=3, label=f'+1 Std: {mean_val + std_val:.2f}')
            axes[i].axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=3, label=f'-1 Std: {mean_val - std_val:.2f}')
            axes[i].set_title(title)
            axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{v}_energy_summary.png'))
        plt.close()

print("Batch processing complete.")