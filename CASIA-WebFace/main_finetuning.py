import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from database import load_dataloaders
from plot import save_train_val_metrics
from utils import get_model, load_checkpoint, fine_tuning_model


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = 224
NUM_WORKERS = 40
PIN_MEMORY = True
EPOCHS = 40
LR = 7e-5
CRITERION = nn.CrossEntropyLoss()
NUM_CLASSES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models to fine-tune
MODELS_TO_EVAL = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# Directory containing model checkpoints
MODELS_PATH_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/checkpoints'

# Directory paths for training, validation, and CSR testing sets
TRAINING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/train'
VALIDATION_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/validation'
CSR_TESTING_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered/close_test'

# Directory to save model results
BASE_RESULTS_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/finetune_evaluation_casia_webface'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Main loop: fine-tune each model
for model_name in MODELS_TO_EVAL:
    print(f'\n{"="*50}\nStarting fine-tuning for model: {model_name}\n{"="*50}')
    
    # Model checkpoint path
    MODEL_PATH = os.path.join(MODELS_PATH_DIR, f'checkpoint_{model_name}', 'best_model.pt')
    checkpoint_dir = os.path.dirname(MODEL_PATH)
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: checkpoint for model {model_name} not found at: {MODEL_PATH}")
    except Exception as e:
        print(f"Error creating/verifying checkpoint directory: {e}")

    # Results directory for this model
    results_dir = os.path.join(BASE_RESULTS_DIR, f'results_{model_name}')
    os.makedirs(results_dir, exist_ok=True)

    # Load training and validation dataloaders
    print('\n\tLoading training and validation dataloaders...')
    train_loader, val_loader, _ = load_dataloaders(IMAGE_SIZE, TRAINING_DIR, VALIDATION_DIR, CSR_TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    print('\tTraining and validation dataloaders loaded.')

    # Load model and pre-trained parameters
    print(f'\n\tLoading model...')
    model = get_model(model_name, num_classes=NUM_CLASSES)
    try:
        model, epoch = load_checkpoint(MODEL_PATH, model, None, None, None, DEVICE)
        print(f'\tModel loaded from: {MODEL_PATH}')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        epoch = 0

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Model parameters info
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n\tTotal model parameters: {total_params:,}')

    # Fine-tuning
    best_model, train_losses, train_accs, val_losses, val_accs = fine_tuning_model(EPOCHS, train_loader, val_loader, DEVICE, model, optimizer, CRITERION)

    # Save best model by validation accuracy
    save_path = os.path.join(results_dir, f'best_model_{model_name}_{NUM_CLASSES}.pth')
    state_dict = best_model.module.state_dict() if isinstance(best_model, nn.DataParallel) else best_model.state_dict()
    torch.save(state_dict, save_path)
    print(f'\n\tBest model saved to: {save_path}')

    # Save training/validation metrics plot
    metrics_png = os.path.join(results_dir, 'fine_tuning_metrics.png')
    save_train_val_metrics(EPOCHS, train_losses, train_accs, val_losses, val_accs, metrics_png)
    print(f'\tMetric plot saved to: {metrics_png}')

    # Cleanup
    torch.cuda.empty_cache()

print(f'\n{"="*50}')
print("Fine-tuning completed for all models.")
print(f"Results saved to: {BASE_RESULTS_DIR}")
print(f'{"="*50}')