import matplotlib.pyplot as plt
import pandas as pd

categories = {
    "Execution Waste": 368,
    "Memory Friction": 328,
    "Boundary Overhead": 25
}


# Create Plot
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(list(categories.keys()), list(categories.values()), color=['#4c72b0', '#55a868', '#c44e52'])

plt.title("Shopizer - Distribution of SonarQube Code Smell Categories", fontsize=16)
plt.ylabel("Number of Issues", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()