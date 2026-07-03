import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt


results = []
versions = ['baseline', 'boundary-overhead', 'execution-waste', 'memory-friction', 'combined']
version_labels = {
    'baseline': 'Baseline',
    'boundary-overhead': 'Boundary Overhead',
    'execution-waste': 'Execution Waste',
    'memory-friction': 'Memory Friction',
    'combined': 'Combined'
}
all_summary_data = []

for v in versions:
    files = glob.glob(f"data/Shopizer/{v}/Locust_*.csv")
    results = []
    for file in files:
        df = pd.read_csv(file)
        agg_row = df[df['Name'] == 'Aggregated']
        if not agg_row.empty:
            results.append({
                'throughput': agg_row['Requests/s'].values[0],
                'latency': agg_row['Average Response Time'].values[0]
            })
    
    df_temp = pd.DataFrame(results)
    stats = df_temp.agg(['mean']).T
    stats['Version'] = v
    all_summary_data.append(stats)

final_df = pd.concat(all_summary_data)


# 1. Pivot the data to make it 'wide'
df_plot = final_df.reset_index().pivot(index='Version', columns='index', values='mean')
df_plot.index = df_plot.index.map(version_labels)
# 2. Setup the dual-axis plot
fig, ax1 = plt.subplots(figsize=(14, 10))
ax2 = ax1.twinx()

# Plot Throughput on left axis
df_plot['throughput'].plot(kind='bar', ax=ax1, color='skyblue', position=1, width=0.3, label='Throughput')
# Plot Latency on right axis
df_plot['latency'].plot(kind='bar', ax=ax2, color='salmon', position=0, width=0.3, label='Latency')

# Formatting
ax1.set_xticklabels(df_plot.index, rotation=0, fontsize=16, fontweight='bold')
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax1.set_xlabel('Versions', fontsize=18, fontweight='bold')
ax1.set_ylabel('Throughput (Requests/s)', fontsize=20, fontweight='bold')
ax2.set_ylabel('Latency (ms)', fontsize=20, fontweight='bold')
ax1.set_title('Shopizer - Performance Metrics by Version', fontsize=20, fontweight='bold')
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15), fontsize=20, frameon=False)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.15), fontsize=20, frameon=False)

plt.tight_layout()  

print(df_plot)
#plt.show()
#plt.savefig('output/Train-Ticket/performance/performance_metrics_train_ticket.png')