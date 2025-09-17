import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

from database import load_dataloaders, load_openset_dataloader
from osr import calculate_weibulls
from utils import get_model, test_open_error_analysis, set_seed

SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
SELECTED_SEED = 43

# Hyperparameters
NUM_WORKERS = 40
PIN_MEMORY = True
BATCH_SIZE = 128
IMAGE_SIZE = 224
NUM_CLASSES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
AVAILABLE_MODELS = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# Human-readable names for plots
MODEL_NAMES_DISPLAY = {
    'inception_v3': 'Inception-V3',
    'resnet50': 'ResNet50',
    'densenet201': 'DenseNet201',
    'mobilenet_v3_large': 'MobileNet-V3-Large',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b_16': 'ViT-B-16'
}

# OpenMax parameters
WEIBULL_TAIL = 5
WEIBULL_ALPHA = 2
WEIBULL_THRESHOLD = 0.9
WEIBULL_DISTANCE = 'euclidean'

# Dataset paths: training, validation, CSR test, OSR test
TRAINING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/train'
VALIDATION_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/validation'
CSR_TESTING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/close_test'
OSR_TESTING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/open_test'

# Directory to save evaluation results
BASE_RESULTS_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

results = {}

# Loop through each model and evaluate
for model_name in AVAILABLE_MODELS:
    print(f'\n{"="*50}\nEvaluating model: {model_name}\n{"="*50}')
    set_seed(SELECTED_SEED)

    # Directory for this model's results
    model_results_dir = os.path.join(BASE_RESULTS_DIR, model_name, f'run_{SEEDS.index(SELECTED_SEED)+1}')
    os.makedirs(model_results_dir, exist_ok=True)

    # Define labels (known classes + "unknown")
    class_names_close = [str(i) for i in range(NUM_CLASSES)]
    class_names_open = class_names_close + ['unknown']

    # Instantiate model
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # Load pretrained weights
    model_path = os.path.join(model_results_dir, f'best_model_{model_name}.pth')
    if not os.path.exists(model_path):
        print(f'\n\tError: model not found at: {model_path}')
        continue

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.module.load_state_dict(new_state_dict)
        else:
            if all(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
        model.eval()
        print(f'\n\tModel loaded from: {model_path}')
    except Exception as e:
        print(f"\n\tError loading model: {e}")
        continue

    # Dataloaders
    train_loader, _, _ = load_dataloaders(IMAGE_SIZE, TRAINING_DIR, VALIDATION_DIR, CSR_TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    opentest_loader = load_openset_dataloader(IMAGE_SIZE, OSR_TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # Fit Weibull distributions for OpenMax
    print('\n\tCalculating Weibull distributions...')
    weibull_models = calculate_weibulls(model, train_loader, DEVICE, NUM_CLASSES, WEIBULL_TAIL, WEIBULL_DISTANCE)
    print('\tWeibull distributions calculated.')

    # Evaluate in the OSR scenario
    print('\n\tEvaluating open-set scenario...')
    labels_true_str, labels_pred_str = test_open_error_analysis(model, opentest_loader, DEVICE, weibull_models,
                                                                NUM_CLASSES, class_names_open, WEIBULL_ALPHA,
                                                                WEIBULL_THRESHOLD, WEIBULL_DISTANCE)
    print('\tOSR evaluation completed.')
    
    results[model_name] = {'true': labels_true_str, 'pred': labels_pred_str}

# Extract ViT-B-16 results for detailed analysis
true_labels = results['vit_b_16']['true']
vit_preds = results['vit_b_16']['pred']

# Separate known and unknown indices
indices_closed = [i for i, label in enumerate(true_labels) if label != 'unknown']
indices_unknown = [i for i, label in enumerate(true_labels) if label == 'unknown']

# Identify correctly classified images by ViT-B-16
indices_closed_vit_correct = [i for i in indices_closed if vit_preds[i] == true_labels[i]]
indices_unknown_vit_correct = [i for i in indices_unknown if vit_preds[i] == true_labels[i]]

# Pick 3 known faces from different identities where ViT is correct
identity_to_indices = defaultdict(list)
for i in indices_closed_vit_correct:
    identity_to_indices[true_labels[i]].append(i)

top3_closed = []
for identity, inds in identity_to_indices.items():
    # Prioritize images where other models disagree with ViT
    sorted_inds = sorted(inds, key=lambda i: sum(results[m]['pred'][i] != true_labels[i] 
                                                 for m in AVAILABLE_MODELS if m != 'vit_b_16'), 
                         reverse=True)
    top3_closed.append(sorted_inds[0])
    if len(top3_closed) == 3:
        break

# Pick 3 unknown images correctly recognized by ViT, where others tend to fail
sorted_unknown = sorted(indices_unknown_vit_correct, key=lambda i: sum(results[m]['pred'][i] != true_labels[i] 
                                                                       for m in AVAILABLE_MODELS if m != 'vit_b_16'), 
                        reverse=True)
top3_unknown = sorted_unknown[:3]

# Create figure (2 rows: known + unknown)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

def add_image_with_text(ax, img_path, true_label, preds):
    
    """ Function to add an image with its true label and predictions from all models. """
    
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')

    # Display true label
    display_label = 'Unknown' if true_label == 'unknown' else f'Identity {true_label}'
    ax.text(0.5, -0.02, f"True Label: {display_label}", 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, weight='bold')

    # Display predictions of all models
    y_offset = -0.12
    for model in AVAILABLE_MODELS:
        pred = preds[model]
        pred_display = 'Unknown' if pred == 'unknown' else f'Identity {pred}'
        symbol, color = ("✔", "green") if pred == true_label else ("✘", "red")
        full_text = f"{MODEL_NAMES_DISPLAY[model]}: {pred_display} {symbol}"
        ax.text(0.5, y_offset, full_text, ha='center', va='top', transform=ax.transAxes, 
                fontsize=14, color=color)
        y_offset -= 0.08

# Add selected known images
for idx, i in enumerate(top3_closed):
    add_image_with_text(axes[idx], opentest_loader.dataset.samples[i][0],
                        true_labels[i],
                        {m: results[m]['pred'][i] for m in AVAILABLE_MODELS})

# Add selected unknown images
for idx, i in enumerate(top3_unknown):
    add_image_with_text(axes[idx + 3], opentest_loader.dataset.samples[i][0],
                        true_labels[i],
                        {m: results[m]['pred'][i] for m in AVAILABLE_MODELS})

plt.subplots_adjust(wspace=0.25, hspace=0.6)
fig_path = os.path.join(BASE_RESULTS_DIR, 'visual_error_analysis.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved at: {fig_path}")