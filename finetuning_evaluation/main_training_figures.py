import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to save the figure
BASE_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'
OUTPUT_FIG = os.path.join(BASE_PATH, 'training_curves_metrics.png')

# List of models
AVAILABLE_MODELS = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# Map for display names
MODEL_NAMES_DISPLAY = {
    'inception_v3': 'Inception-V3',
    'resnet50': 'ResNet50',
    'densenet201': 'DenseNet201',
    'mobilenet_v3_large': 'MobileNet-V3-Large',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b_16': 'ViT-B-16'
}

# Load all data
all_data = []
for model in os.listdir(BASE_PATH):
    model_dir = os.path.join(BASE_PATH, model)
    if not os.path.isdir(model_dir):
        continue
    for run in os.listdir(model_dir):
        run_dir = os.path.join(model_dir, run)
        csv_path = os.path.join(run_dir, 'fine_tuning_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Model"] = model
            df["Run"] = run
            all_data.append(df)

if not all_data:
    raise ValueError('No fine_tuning_metrics.csv files were found in the results folders.')

df_all = pd.concat(all_data, ignore_index=True)

# Metrics to plot
metrics = [
    ("train_loss", "val_loss", "Loss"),
    ("train_r", "val_r", "Recall"),
    ("train_p", "val_p", "Precision")
]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharex=True)

for idx, (train_m, val_m, label) in enumerate(metrics):
    ax = axes[idx]

    for model in AVAILABLE_MODELS:
        df_model = df_all[df_all["Model"] == model]
        if df_model.empty:
            continue

        # Mean per epoch
        grouped = df_model.groupby("epoch")[[train_m, val_m]].mean()
        epochs = grouped.index

        # Training curve
        ax.plot(epochs, grouped[train_m], "--", label=f"{MODEL_NAMES_DISPLAY[model]} (Train)")
        # Validation curve
        ax.plot(epochs, grouped[val_m], label=f"{MODEL_NAMES_DISPLAY[model]} (Val)")

    ax.set_xlabel("Epochs", fontsize=16)
    if label in ["Recall", "Precision"]:
        ax.set_ylabel(f"{label} (%)", fontsize=16)
    else:
        ax.set_ylabel(label, fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)

# Personalized legend
handles, labels = [], []
for model in AVAILABLE_MODELS:
    for line in axes[0].get_lines():
        if line.get_label().startswith(MODEL_NAMES_DISPLAY[model]):
            handles.append(line)
            labels.append(line.get_label())

fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.3), fontsize=16)

# Save figure
plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches="tight", dpi=300)
plt.close()

print(f'Figure exported as {OUTPUT_FIG}')
