import os
import random
import numpy as np

import torch
import torch.nn as nn

from database import load_dataloaders, load_openset_dataloader
from plot import plot_roc_curve
from osr import calculate_weibulls
from utils import get_model, test_close, test_open


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
NUM_WORKERS = 40
PIN_MEMORY = True
BATCH_SIZE = 128
IMAGE_SIZE = 224
NUM_CLASSES = 100
NUM_IMAGES_UNKNOWN = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models to evaluate
AVAILABLE_MODELS = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# OpenMax parameters
WEIBULL_TAIL = 5
WEIBULL_ALPHA = 2
WEIBULL_THRESHOLD = 0.9
WEIBULL_DISTANCE = 'euclidean'

# Directory paths for training, validation and CSR/OSR testing sets
TRAINING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/train'
VALIDATION_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/validation'
CSR_TESTING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/close_test'
OSR_TESTING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/open_test'

# Directory to save model results
BASE_RESULTS_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/finetune_evaluation_casia_webface'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Main loop: evaluation of each model in CSR and OSR scenarios
for model_name in AVAILABLE_MODELS:
    print(f'\n{"="*50}\n Evaluating model: {model_name}\n{"="*50}')
    
    # Directory for this model's results
    model_results_dir = os.path.join(BASE_RESULTS_DIR, f'results_{model_name}')
    os.makedirs(model_results_dir, exist_ok=True)

    # Labels for closed and open set datasets
    class_names_close = [str(i) for i in range(NUM_CLASSES)]
    class_names_open = class_names_close + ['unknown']  # Add 'unknown' class for open set

    # Model creation
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load fine-tuned model weights
    model_path = os.path.join(model_results_dir, f'best_model_{model_name}_{NUM_CLASSES}.pth')
    if not os.path.exists(model_path):
        print(f"\tError: model not found at: {model_path}")
        continue  # Skip to the next model if checkpoint is missing

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        # Adjust keys if the model was saved using DataParallel
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

    # Load training and closed set testing datasets
    train_loader, _, test_loader = load_dataloaders(IMAGE_SIZE, TRAINING_DIR, VALIDATION_DIR, CSR_TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # Loss function for evaluation
    criterion = nn.CrossEntropyLoss()

    # Evaluation on closed set
    print('\n\tEvaluating on CSR scenario...')
    test_close(model, DEVICE, test_loader, criterion, class_names_close)
    print('\tEvaluation completed.')

    # Compute Weibull distributions for OpenMax
    print('\n\tCalculating Weibull distributions...')
    weibull_models = calculate_weibulls(model, train_loader, DEVICE, NUM_CLASSES, WEIBULL_TAIL, WEIBULL_DISTANCE)
    print('\tWeibull distributions calculated.')

    # Load open set testing dataset
    print('\n\tLoading open set dataloader...')
    opentest_loader = load_openset_dataloader(IMAGE_SIZE, OSR_TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    print('\tOpen set dataloader loaded.')

    # Evaluation on open set
    print('\n\tEvaluating on OSR scenario...')
    labels_open, scores_unknown = test_open(model, opentest_loader, DEVICE, weibull_models, NUM_CLASSES, class_names_open, WEIBULL_ALPHA, WEIBULL_THRESHOLD, WEIBULL_DISTANCE)
    print('\tEvaluation completed.')

    # Save ROC curve
    plot_roc_curve(labels_open, scores_unknown, NUM_CLASSES, os.path.join(model_results_dir, f'roc_curve_{model_name}_{NUM_CLASSES}_{NUM_IMAGES_UNKNOWN}.png'))
    print(f'\n\tROC curve saved at: {model_results_dir}')

    # Cleanup
    torch.cuda.empty_cache()
    print(f"\tMemory cleared after evaluating {model_name}")

print(f"\n{'='*50}")
print(f"Evaluation completed for all selected models.")
print(f"Results saved in: {BASE_RESULTS_DIR}")
print(f"{'='*50}")
