import os
import pandas as pd

import torch
import torch.nn as nn

from database import load_dataloaders, load_openset_dataloader
from osr import calculate_weibulls
from utils import get_model, test_close, test_open_softmax_thresholding, set_seed

SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

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

# OpenMax parameters (for SoftMax Thresholding, we only use weibull_threshold parameter, but the rest of the parameters are kept because the code needs them)
WEIBULL_TAIL = 5
WEIBULL_ALPHA = 2
WEIBULL_THRESHOLD = 0.9
WEIBULL_DISTANCE = 'euclidean'

# Paths for training, validation and testing (CSR and OSR) datasets
TRAINING_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/train'
VALIDATION_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/validation'
CSR_TESTING_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/close_test'
OSR_TESTING_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface_imgs_mtcnn_ordered/open_test'

# Path to save results
RESULTS_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'
os.makedirs(RESULTS_PATH, exist_ok=True)

# Main loop: evaluation of each model in CSR and OSR scenarios with multiple seeds
for model_name in AVAILABLE_MODELS:
    print(f'\n{"="*50}\n Evaluating model: {model_name}\n{"="*50}')
        
    for run_idx, seed in enumerate(SEEDS):
        print(f'\n\t===> Run {run_idx+1} with seed {seed} <===')
        set_seed(seed)
    
        # Path to save the results of this specific model and run
        model_run_results_path = os.path.join(RESULTS_PATH, model_name, f'run_{run_idx+1}')
        os.makedirs(model_run_results_path, exist_ok=True)

        # Labels for CSR and OSR testing datasets
        class_names_close = [str(i) for i in range(NUM_CLASSES)]
        class_names_open = class_names_close + ['unknown']  # Add 'unknown' class for OSR scenario

        # Model creation
        model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # Load fine-tuned model weights
        model_path = os.path.join(model_run_results_path, f'best_model_{model_name}.pth')
        if not os.path.exists(model_path):
            print(f'\t\tError: model not found at: {model_path}')
            continue  # Skip

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
            print(f'\n\t\tModel loaded from: {model_path}')
        except Exception as e:
            print(f'\n\t\tError loading model: {e}')
            continue # Skip

        # Load training and testing (CSR) datasets
        print('\n\t\tLoading training and testing (CSR) dataloader...')
        train_loader, _, test_loader = load_dataloaders(IMAGE_SIZE, TRAINING_PATH, VALIDATION_PATH, CSR_TESTING_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        print('\t\tTraining and testing (CSR) dataloader loaded.')
        
        # Loss function for evaluation
        criterion = nn.CrossEntropyLoss()

        # CSR scenario evaluation
        print('\n\t\tEvaluating on CSR scenario...')
        csr_metrics = test_close(model, DEVICE, test_loader, criterion, class_names_close)
        print('\t\tEvaluation completed.')

        # Compute Weibull distributions for OpenMax
        print('\n\t\tCalculating Weibull distributions...')
        weibull_models = calculate_weibulls(model, train_loader, DEVICE, NUM_CLASSES, WEIBULL_TAIL, WEIBULL_DISTANCE)
        print('\t\tWeibull distributions calculated.')

        # Load testing (OSR) dataset
        print('\n\t\tLoading testing (OSR) dataloader...')
        opentest_loader = load_openset_dataloader(IMAGE_SIZE, OSR_TESTING_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        print('\t\tTesting (OSR) dataloader loaded.')

        # OSR scenario evaluation
        print('\n\t\tEvaluating on OSR scenario...')
        osr_metrics = test_open_softmax_thresholding(model, opentest_loader, DEVICE, weibull_models, NUM_CLASSES, class_names_open, WEIBULL_ALPHA, WEIBULL_THRESHOLD, WEIBULL_DISTANCE)
        print('\t\tEvaluation completed.')
        
        # Save metrics in CSV
        results = {
            "csr_f1": csr_metrics["f1_macro"],
            "osr_f1": osr_metrics["f1_macro"],
            "osr_fpr": osr_metrics["fpr"],
            "osr_tnr": osr_metrics["tnr"],
            "osr_tpr": osr_metrics["tpr"],
            "osr_fnr": osr_metrics["fnr"]
        }
        csv_path = os.path.join(model_run_results_path, 'evaluation_metrics_softmax_thresholding.csv')
        df = pd.DataFrame([results])
        df.to_csv(csv_path, index=False)
        print(f'\t\tMetrics CSV saved at: {csv_path}')

        # Clean up
        torch.cuda.empty_cache()

print(f'\n{"="*50}')
print(f'Evaluation completed for all selected models.')
print(f'Results saved in: {RESULTS_PATH}')
print(f'{"="*50}')
