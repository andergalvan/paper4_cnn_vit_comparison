import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts

from torchvision import datasets, transforms

from utils import get_model, train_one_epoch, validate, save_checkpoint, setup_tensorboard


# Variable parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help="Model name.")
parser.add_argument('--log_dir', type=str, required=True, help="Where to save the model logs.")
parser.add_argument('--checkpoint_dir', type=str, required=True, help="Where to save the model checkpoints.")
args = parser.parse_args()

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# VGGFace2 training set directory
DATA_DIR = '/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/vggface2/train'

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6
NUM_CLASSES = 8631
NUM_WORKERS = 40
USE_AMP = True
ACCUMULATION_STEPS = 1
PATIENCE = 5
WARM_UP_EPOCHS = 5

try:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    print(f"[INFO] USING DEVICE: {device}, LOCAL RANK: {local_rank}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = setup_tensorboard(args.log_dir) if local_rank == 0 else None

    # Training and validation transformations
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Load training set
    full_dataset = datasets.ImageFolder(DATA_DIR)

    # Divide the training set into 95% training and 5% validation subsets
    val_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply the transformations
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    print(f"[DEBUG 5] DATASET DE ENTRENAMIENTO Y VALIDACIÓN CREADOS")  

    # Creation of the distributed samplers for training and validation
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True, drop_last=True, seed=42)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)

    # Creation of the dataloaders with the samplers
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # Model initialization
    model = get_model(args.model_name, NUM_CLASSES)
    print(f"[INFO] MODEL {args.model_name} LOADED.")
    print("[DEBUG 6] MODELO CREADO") 
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Schedulers (WarmUp + CosineAnnealingWarmRestarts)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARM_UP_EPOCHS)
    T_0 = max(EPOCHS - WARM_UP_EPOCHS, 1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=MIN_LR)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, scheduler], milestones=[WARM_UP_EPOCHS])

    best_val_acc = 0.0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Main training loop
    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        print(f"\n[INFO] EPOCH {epoch}/{EPOCHS}")

        # Training of an epoch
        train_loss, train_acc, train_acc_top5 = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, USE_AMP, ACCUMULATION_STEPS,
                                                                writer=writer, epoch=epoch)
        
        # Validation of an epoch
        val_loss, val_acc, val_acc_top5 = validate(model, val_loader, criterion, device, USE_AMP)
        early_stop = torch.tensor(0, device=device)

        # Early stopping
        if local_rank == 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0

                # Save the best model
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                checkpoint_data = {'model_state_dict': model.module.state_dict()}
                save_checkpoint(checkpoint_data, save_path, epoch, optimizer, scheduler=combined_scheduler, scaler=scaler)
                print(f"[INFO] MEJOR MODELO GUARDADO EN: {save_path}")
            else:
                epochs_no_improve += 1
                print(f"[INFO] EPOCHS SIN MEJORA: {epochs_no_improve}/{PATIENCE}")

            if epochs_no_improve >= PATIENCE:
                early_stop.fill_(1)   # Signal to stop training
                
        # Broadcast to all training completion processes
        dist.broadcast(early_stop, src=0)

        if early_stop.item() == 1:
            if local_rank == 0:
                print(f"[INFO] DETENIDO EL ENTRENAMIENTO POR EARLY STOPPING EN LA ÉPOCA {epoch}.")
            break

        # Writing metrics in TensorBoard
        if writer and local_rank == 0:
            writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
            writer.add_scalars('Accuracy', {'Train Top-1': train_acc, 'Val Top-1': val_acc, 'Val Top-5': val_acc_top5}, epoch)

        print(f"\nEPOCH {epoch}/{EPOCHS}")
        print(f"TRAIN LOSS: {train_loss:.4f} | TOP-1: {train_acc:.2f}% | TOP-5: {train_acc_top5:.2f}%")
        print(f"VAL LOSS:   {val_loss:.4f} | TOP-1: {val_acc:.2f}% | TOP-5: {val_acc_top5:.2f}%")

        # Checkpoint saving every 5 epochs
        if epoch % 5 == 0 and local_rank == 0:
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            checkpoint_data = {'model_state_dict': model.module.state_dict()}
            save_checkpoint(checkpoint_data, save_path, epoch, optimizer, scheduler=combined_scheduler, scaler=scaler)
            print(f"[INFO] CHECKPOINT GUARDADO EN ÉPOCA {epoch}: {save_path}")

        # Scheduler update
        combined_scheduler.step()

    # Closing the TensorBoard writer
    if writer and local_rank == 0:
        writer.close()
    print("[INFO] ENTRENAMIENTO COMPLETADO.")
    dist.destroy_process_group()

except Exception as e:
    print(f"[CRASH] ERROR: {str(e)}")
    raise
