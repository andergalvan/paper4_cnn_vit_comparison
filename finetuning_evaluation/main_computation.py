import time
import os
import numpy as np
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

import torch
import torch.nn as nn

from utils import get_model, set_seed

SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

# Available model identifiers to evaluate
AVAILABLE_MODELS = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# Human-friendly names used for plotting
MODEL_NAMES_DISPLAY = {
    'inception_v3': 'Inception-V3',
    'resnet50': 'ResNet50',
    'densenet201': 'DenseNet201',
    'mobilenet_v3_large': 'MobileNet-V3-Large',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b_16': 'ViT-B-16'
}

# Hyperparameters 
IMAGE_SIZE = 224
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory where result figure will be saved; create if missing
BASE_RESULTS_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Containers to collect metrics for plotting
model_names = []
flops_list = []
time_mean_list = []
time_std_list = []

for model_name in AVAILABLE_MODELS:
    print(f"\n{'='*50}\n Evaluating model: {model_name}\n{'='*50}")

    # Instantiate model (no pretrained weights)
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    # Compute and print total number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

    # Move model to CPU temporarily to compute FLOPs (MACs) using ptflops
    model_cpu = model.to("cpu")
    macs, _ = get_model_complexity_info(model_cpu, (3, IMAGE_SIZE, IMAGE_SIZE), backend='aten', as_strings=False, print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {macs/1e9:.2f} GMac")
    flops_list.append(macs)

    # If multiple GPUs available, wrap in DataParallel for inference
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE).eval()

    # Measure inference time across multiple random seeds for stability
    times_seeds = []
    for seed in SEEDS:
        set_seed(seed)
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # Measurement
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy_input)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        times_seeds.append(np.mean(times))

    # Aggregate across seeds and convert to milliseconds
    times_seeds = np.array(times_seeds)
    mean_time = 1000 * times_seeds.mean() # mean latency in ms
    std_time = 1000 * times_seeds.std() # std dev in ms
    print(f"Average inference time: {mean_time:.2f} ± {std_time:.2f} ms")

    time_mean_list.append(mean_time)
    time_std_list.append(std_time)
    model_names.append(model_name)

# Prepare display names and x positions for bars
display_names = [MODEL_NAMES_DISPLAY[m] for m in model_names]
x = np.arange(len(model_names))
width = 0.6

# Create side-by-side bar plots: FLOPs and Inference Time (with error bars)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,7))

# FLOPs plot (convert MACs to GMac)
ax1.bar(x, [f/1e9 for f in flops_list], width, color='skyblue')
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=16)
ax1.set_ylabel("FLOPs (GMac)", fontsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
max_flops = max([f/1e9 for f in flops_list])
ax1.set_ylim(0, max_flops*1.2)
for i, f in enumerate(flops_list):
    ax1.text(i, f/1e9 + 0.05*max_flops, f"{f/1e9:.2f}", ha='center', va='bottom', fontsize=16)

# Inference time plot with error bars (mean ± std)
ax2.bar(x, time_mean_list, width, color='lightgreen', yerr=time_std_list, capsize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(display_names, rotation=45, ha='right', fontsize=16)
ax2.set_ylabel("Inference Time (ms)", fontsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
max_time = max([mean+std for mean, std in zip(time_mean_list, time_std_list)])
ax2.set_ylim(0, max_time*1.2)
for i, (mean, std) in enumerate(zip(time_mean_list, time_std_list)):
    ax2.text(i, mean + std + 0.02*max_time, f"{mean:.1f}±{std:.1f}", ha='center', va='bottom', fontsize=16)

plt.tight_layout()
fig_path = os.path.join(BASE_RESULTS_DIR, 'computation.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFigure saved at: {fig_path}")