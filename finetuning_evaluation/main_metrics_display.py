import matplotlib.pyplot as plt
import numpy as np
import os

# Path where results are stored
RESULTS_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'

# Metrics to plot
metrics_to_plot = ["TPR", "FPR", "TNR", "FNR"]
num_vars = len(metrics_to_plot)

# Angles for the radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Mean values for each model (in %)
models_data = {
    "Inception-V3": {"TPR": 99.70, "FPR": 36.98, "TNR": 63.02, "FNR": 0.30, "F1": 74.39},
    "ResNet50": {"TPR": 99.23, "FPR": 22.37, "TNR": 77.63, "FNR": 0.77, "F1": 85.96},
    "DenseNet201": {"TPR": 99.45, "FPR": 24.16, "TNR": 75.84, "FNR": 0.55, "F1": 84.78},
    "MobileNet-V3-Large": {"TPR": 91.97, "FPR": 15.19, "TNR": 84.81, "FNR": 8.03, "F1": 87.32},
    "EfficientNet-B0": {"TPR": 98.42, "FPR": 19.35, "TNR": 80.65, "FNR": 1.58, "F1": 87.65},
    "ViT-B-16": {"TPR": 98.50, "FPR": 15.08, "TNR": 84.92, "FNR": 1.50, "F1": 90.50}
}

# Assign colors for consistency
colors = {
    "Inception-V3": "#1f77b4",
    "ResNet50": "#ff7f0e",
    "DenseNet201": "#2ca02c",
    "MobileNet-V3-Large": "#d62728",
    "EfficientNet-B0": "#9467bd",
    "ViT-B-16": "#8c564b"
}

# Create figure
fig, ax1 = plt.subplots(figsize=(13, 8), subplot_kw=dict(polar=True))

# Radar chart
for model, metrics in models_data.items():
    values = [metrics[m] for m in metrics_to_plot]
    values += values[:1]  # close the circle
    ax1.plot(angles, values, linewidth=2, label=model, color=colors[model])
    ax1.fill(angles, values, alpha=0.25, color=colors[model])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metrics_to_plot, fontsize=16)
ax1.set_yticklabels([])

# Legend with F1 in parentheses
handles = [plt.Line2D([0], [0], color=colors[m], lw=2) for m in models_data.keys()]
labels = [f"{m} (F1={metrics['F1']:.2f}%)" for m, metrics in models_data.items()]
fig.legend(handles, labels, loc='center right', fontsize=16, bbox_to_anchor=(1.25, 0.5))

# Save figure
fig_path = os.path.join(RESULTS_PATH, "visual_comparative.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved at: {fig_path}")
