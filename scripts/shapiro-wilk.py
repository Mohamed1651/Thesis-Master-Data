import pandas as pd
import scipy.stats as stats
import os
import re

# Dictionary to hold data: data_store[project][version] = dataframe
data_store = {}
projects = ['Train-Ticket']

# ... (Insert your extract_metrics function here) ...
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



for project in projects:
    data_store[project] = {}
    base_path = os.path.join('../data', project)
    versions = ['baseline' ,'execution-waste', 'memory-friction', 'combined']
    
    for v in versions:
        v_path = os.path.join(base_path, v)
        run_data = []
        files = [f for f in os.listdir(v_path) if re.match(r'^data\d+\.csv$', f)]
        
        for filename in files:
            stats_data = extract_metrics(os.path.join(v_path, filename))
            if stats_data:
                run_data.append(stats_data)
        
        if run_data:
            data_store[project][v] = pd.DataFrame(run_data)

# --- Statistical Testing per Project ---
for project in data_store:
    print(f"\n=== Statistical Tests for {project} ===")
    
    print("\n--- Shapiro-Wilk Test (Normality) ---")
    for v, df in data_store[project].items():
        _, p_cpu = stats.shapiro(df['CPU Energy (J)'])
        _, p_dram = stats.shapiro(df['DRAM Energy (J)'])
        print(f"{v.capitalize()} | CPU p-value: {p_cpu:.4f} | DRAM p-value: {p_dram:.4f}")
        