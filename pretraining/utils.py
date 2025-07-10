from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet50, mobilenet_v3_large, inception_v3, vit_b_16, densenet201, efficientnet_b0


def setup_tensorboard(log_dir):
    return SummaryWriter(log_dir)


def get_model(model_name, num_classes, pretrained=False):
    
    """ Creates and returns the model according to the specified name. """

    if model_name == 'vit_b_16':
        model = vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features

    elif model_name == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features

    elif model_name == 'mobilenet_v3_large':
        model = mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features

    elif model_name == 'inception_v3':
        model = inception_v3(weights='IMAGENET1K_V1' if pretrained else None, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features
        
    elif model_name == 'densenet201':
        model = densenet201(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features
        
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.get_embedding_dim = lambda: in_features

    else:
        raise ValueError(f"Model not supported: {model_name}")

    return model



def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp, accumulation_steps, writer=None, epoch=None):
    
    """ Training the model for one epoch. """
    
    model.train()
    epoch_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs)
            original_loss = criterion(logits, targets)
        
        loss = original_loss/accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += original_loss.item() * inputs.size(0) 
        acc1, acc5 = calculate_accuracy(logits, targets, topk=(1, 5))
        top1_correct += acc1.item() * inputs.size(0) / 100
        top5_correct += acc5.item() * inputs.size(0) / 100
        total += targets.size(0)

    top1_correct = reduce_value(torch.tensor(top1_correct, device=device), average=False)
    top5_correct = reduce_value(torch.tensor(top5_correct, device=device), average=False)
    total_tensor = reduce_value(torch.tensor(total, device=device), average=False)

    epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
    epoch_loss = reduce_value(epoch_loss_tensor, average=False)
    epoch_loss = epoch_loss / total_tensor.float()
    top1_acc = 100. * top1_correct.float() / total_tensor.float()
    top5_acc = 100. * top5_correct.float() / total_tensor.float()

    return epoch_loss, top1_acc, top5_acc


def validate(model, dataloader, criterion, device, use_amp):
    
    """ Validar el modelo durante una época. """
    
    model.eval()
    val_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            val_loss += loss.item() * inputs.size(0)
            acc1, acc5 = calculate_accuracy(logits, targets, topk=(1, 5))
            top1_correct += acc1.item() * inputs.size(0) / 100
            top5_correct += acc5.item() * inputs.size(0) / 100
            total += targets.size(0)

    top1_correct = reduce_value(torch.tensor(top1_correct, device=device), average=False)
    top5_correct = reduce_value(torch.tensor(top5_correct, device=device), average=False)
    total_tensor = reduce_value(torch.tensor(total, device=device), average=False)

    val_loss = reduce_value(torch.tensor(val_loss, device=device),average=False)
    val_loss = val_loss / total_tensor.float()  # Normaliza la pérdida por el número total de muestras
    top1_acc = 100. * top1_correct.float() / total_tensor.float()
    top5_acc = 100. * top5_correct.float() / total_tensor.float()

    return val_loss, top1_acc, top5_acc


def reduce_value(value, average=True):
    
    """ Synchronizes and reduces a numerical value across multiple GPUs in a distributed environment. """
    
    device = torch.cuda.current_device()
    
    if not isinstance(value, torch.Tensor):
        value_tensor = torch.tensor(value, dtype=torch.float32, device=device)
    else:
        value_tensor = value.to(device)

    world_size = dist.get_world_size()
    if world_size > 1:
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        if average:
            value_tensor /= world_size
    
    return value_tensor 


def save_checkpoint(state_dict, path, epoch=None, optimizer=None, scheduler=None, scaler=None):
    
    """ Save checkpoint. """
    
    torch.save({
        'model_state_dict': state_dict['model_state_dict'],
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None
    }, path)


def calculate_accuracy(logits, targets, topk=(1,)):
    
    """ Calculates top-k accuracy. """
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res