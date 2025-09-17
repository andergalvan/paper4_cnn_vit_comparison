import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from database import load_dataloaders
from plot import save_train_val_metrics
from utils import get_model, load_checkpoint, fine_tuning_model, set_seed

SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

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

# Path containing models checkpoints from pre-training phase
PRETRAINED_MODELS_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/checkpoints'

# Paths for training, validation, and testing (CSR) sets
TRAINING_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/train'
VALIDATION_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/validation'
CSR_TESTING_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/close_test'

# Path to save results
RESULTS_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'
os.makedirs(RESULTS_PATH, exist_ok=True)

# Main loop: fine-tune each model with multiple seeds
for model_name in MODELS_TO_EVAL:
    print(f'\n{"="*50}\nStarting fine-tuning for model: {model_name}\n{"="*50}')
    
    for run_idx, seed in enumerate(SEEDS):
        print(f'\n\t===> Run {run_idx+1} with seed {seed} <===')
        set_seed(seed)
        
        # Path to save the results of this specific model and run
        model_run_results_path = os.path.join(RESULTS_PATH, f'{model_name}', f'run_{run_idx+1}')
        os.makedirs(model_run_results_path, exist_ok=True)
    
        # Pre-trained model path 
        model_path = os.path.join(PRETRAINED_MODELS_PATH, model_name, 'best_model.pt')
        checkpoint_path = os.path.dirname(model_path)
        try:
            os.makedirs(checkpoint_path, exist_ok=True)
            if not os.path.exists(model_path):
                print(f'\n\t\tWarning: pre-training checkpoint for model {model_name} not found at: {model_path}')
        except Exception as e:
            print(f'\n\t\tError creating/verifying pre-training checkpoint path: {e}')

        # Load training and validation dataloaders
        print('\n\t\tLoading training and validation dataloaders...')
        train_loader, val_loader, _ = load_dataloaders(IMAGE_SIZE, TRAINING_PATH, VALIDATION_PATH, CSR_TESTING_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        print('\t\tTraining and validation dataloaders loaded.')

        # Load model and pre-trained weights
        print(f'\n\t\tLoading model...')
        model = get_model(model_name, num_classes=NUM_CLASSES)
        try:
            model, epoch = load_checkpoint(model_path, model, None, None, None, DEVICE)
            print(f'\t\tModel loaded from: {model_path}')
        except Exception as e:
            print(f'\t\tError loading pre-training checkpoint: {e}')
            epoch = 0

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        model = model.to(DEVICE)
        
        # Model parameters info
        total_params = sum(p.numel() for p in model.parameters())
        print(f'\n\t\tTotal model parameters: {total_params:,}')

        # Fine-tuning
        metrics = fine_tuning_model(EPOCHS, train_loader, val_loader, DEVICE, model, optimizer, CRITERION) # R --> Recall; P --> Precision
        best_model = metrics['best_model']
        train_losses = metrics['train_losses']
        train_accuracies = metrics['train_accuracies']
        train_rvalues = metrics['train_rvalues']
        train_pvalues = metrics['train_pvalues']
        val_losses = metrics['val_losses']
        val_accuracies = metrics['val_accuracies']
        val_rvalues = metrics['val_rvalues']
        val_pvalues = metrics['val_pvalues']

        # Save best model by validation accuracy
        save_path = os.path.join(model_run_results_path, f'best_model_{model_name}.pth')
        state_dict = best_model.module.state_dict() if isinstance(best_model, nn.DataParallel) else best_model.state_dict()
        torch.save(state_dict, save_path)
        print(f'\n\t\tBest model saved to: {save_path}')

        # Save training/validation metrics plot
        metrics_png = os.path.join(model_run_results_path, 'fine_tuning_metrics.png')
        save_train_val_metrics(EPOCHS, train_losses, train_rvalues, train_pvalues, val_losses, val_rvalues, val_pvalues, metrics_png)
        print(f'\t\tMetrics plot saved to: {metrics_png}')
        
        # Save training/validation metrics in CSV
        df = pd.DataFrame({
            'epoch': list(range(1, EPOCHS + 1)),
            'train_loss': train_losses,
            'train_r': train_rvalues,
            'train_p': train_pvalues,
            'val_loss': val_losses,
            'val_r': val_rvalues,
            'val_p': val_pvalues
        })
        csv_path = os.path.join(model_run_results_path, f'fine_tuning_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f'\t\tMetrics CSV saved to: {csv_path}')
        
        # Clean up
        torch.cuda.empty_cache()

print(f'\n{"="*50}')
print('Fine-tuning completed for all models.')
print(f'Results saved to: {RESULTS_PATH}')
print(f'{"="*50}')