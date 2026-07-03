import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'figure.dpi': 300})

versions = ['baseline', 'combined', 'execution-waste', 'memory-friction']
plt.figure(figsize=(6, 4))

for v in versions:
    files = glob.glob(f'data/Train-Ticket/{v}/data*.csv')
    
    all_runs = []
    
    for file_path in files:
        df = pd.read_csv(file_path)
        cpu_cols = [c for c in df.columns if 'CPU_USAGE' in c]
        avg_cpu = df[cpu_cols].mean(axis=1)
        # Calculate time in seconds
        time_sec = df['Delta'].cumsum() / 1000
        # Create a combined series with time as index
        run_data = pd.Series(avg_cpu.values, index=time_sec)
        all_runs.append(run_data)
    
    common_time = np.arange(0, max(run.index.max() for run in all_runs), 1)
    resampled_runs = [run.reindex(common_time, method='ffill') for run in all_runs]
    print(f"Processed {len(resampled_runs)} runs for version: {v}")
    mean_cpu = pd.concat(resampled_runs, axis=1).mean(axis=1)
    
    plt.plot(mean_cpu.index, mean_cpu.values, label=v, linewidth=1.5)

plt.title('Train-Ticket - Mean CPU Utilization per Version', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Average CPU Usage (%)')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig('output/Train-Ticket/cpu-usage/cpu_usage.png')