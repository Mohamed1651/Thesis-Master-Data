import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'figure.dpi': 300})

files = glob.glob('data/Shopizer/memory-friction/data*.csv')
files.sort() 

for file_path in files:
    df = pd.read_csv(file_path)
    
    cpu_cols = [c for c in df.columns if 'CPU_USAGE' in c]
    avg_cpu = df[cpu_cols].mean(axis=1)
    df['Elapsed_Delta'] = df['Delta'].cumsum()
    
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    elapsed_seconds = df['Elapsed_Delta'] / 1000
    ax.plot(elapsed_seconds, avg_cpu, color='steelblue', linewidth=1)
    
    file_name = os.path.basename(file_path)
    ax.set_title(f'CPU Utilization: {file_name}', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Usage (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    output_path = f'output/Shopizer/cpu-usage/memory-friction/{file_name.replace(".csv", ".png")}'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Processed: {file_name}")