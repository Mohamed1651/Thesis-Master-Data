import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import os
import re

PROJECT = 'Sock-Shop'
VERSIONS = ['baseline', 'execution-waste', 'memory-friction', 'boundary-overhead' ,'combined']
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


data_store = {}
for v in VERSIONS:
    v_path = os.path.join('../data', PROJECT, v)
    run_data = []
    if os.path.exists(v_path):
        for filename in [f for f in os.listdir(v_path) if re.match(r'^data\d+\.csv$', f)]:
            metrics = extract_metrics(os.path.join(v_path, filename))
            if metrics is not None:
                metrics['version'] = v
                run_data.append(metrics)
    
    if run_data:
        data_store[v] = pd.DataFrame(run_data)

if not data_store:
    print("No valid data loaded. Check your folder structure/column names.")
else:
    full_df = pd.concat(data_store.values(), ignore_index=True)
    print(full_df.head(100000)) 

    for metric in ['CPU Energy (J)', 'DRAM Energy (J)']:
        print(f"\nMetric: {metric}")
        groups = [group[metric] for name, group in full_df.groupby('version')]
        h_stat, p_val = stats.kruskal(*groups)
        print(f"Kruskal-Wallis p-value: {p_val:.10f}")