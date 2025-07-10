import numpy as np
import copy

import torch
import torch.nn as nn

from torchvision.models import resnet50, mobilenet_v3_large, inception_v3, vit_b_16, densenet201, efficientnet_b0

from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score

from evaluation import *
from openmax import *
from plot import *


def get_model(model_name, num_classes, pretrained=False):
    
    """ Creates and returns a model based on the specified name. """

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
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def load_checkpoint(path, model, optimizer, scheduler, scaler, device, strict=False):
    
    """Load pre-trained model and return model and last training epoch."""

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix if present (for DataParallel compatibility)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Filter incompatible keys (e.g. classifier layer: 8631 --> 100)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return model, checkpoint.get('epoch', 0)


def fine_tuning_model(epochs, train_loader, val_loader, device, model, optimizer, criterion):
    
    """Fine-tune the model (training and validation)."""

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Display epoch metrics
        print(
            f'\n\tEpoch [{epoch+1}/{epochs}] - '
            f'Train Loss: {train_loss:.4f} ({train_accuracy:.2f}%), '
            f'Val Loss: {val_loss:.4f} ({val_accuracy:.2f}%)'
        )

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = copy.deepcopy(model)
            print(f'\t\t✅ Best model updated: Val Acc = {best_val_acc:.2f}%')

    return best_model, train_losses, train_accuracies, val_losses, val_accuracies


def test_close(model, device, test_loader, criterion, class_names_close):
    
    """Evaluation on CSR scenario."""

    model.eval()

    test_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    test_f1 = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_report = classification_report(all_labels, all_preds, target_names=class_names_close)
    test_precision = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_recall = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_accuracy = 100 * accuracy_score(all_labels, all_preds)

    print('\n\t\tF1 Macro: {:.2f}%'.format(test_f1))
    print('\t\tAccuracy: {:.2f}%'.format(test_accuracy))
    print('\t\tPrecision: {:.2f}%'.format(test_precision))
    print('\t\tRecall: {:.2f}%'.format(test_recall))
    print('\t\tClassification Report:\n', test_report)


def test_open(net, opentest_loader, device, weibull_model, num_classes, target_names, weibull_alpha, weibull_threshold, weibull_distance): 
    
    """Evaluation on OSR scenario using OpenMax."""

    scores, labels = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(opentest_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            scores.append(outputs)
            labels.append(targets)

    # Convert tensors to numpy arrays
    scores = torch.cat(scores, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # OpenMax predictions
    pred_openmax = []
    openmax_probs = []

    for score in scores:
        categories = list(range(0, num_classes))
        so, ss = openmax(weibull_model, categories, score, 0.5, weibull_alpha, weibull_distance)
        pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else num_classes)
        openmax_probs.append(so)

    pred_openmax_np = np.array(pred_openmax)
    openmax_probs_np = np.array(openmax_probs)

    # Binary labels (1: unknown, 0: known)
    binary_labels = (labels == num_classes).astype(int)
    binary_preds = (pred_openmax_np == num_classes).astype(int)

    # Confusion matrix
    cm_binary = confusion_matrix(binary_labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_binary[0, 0], cm_binary[0, 1], cm_binary[1, 0], cm_binary[1, 1]

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f'\n\t\tFalse Positive Rate (FPR): {100 * fpr:.2f}%')
    print(f'\t\tTrue Negative Rate (TNR): {100 * tnr:.2f}%')
    print(f'\t\tTrue Positive Rate (TPR): {100 * tpr:.2f}%')
    print(f'\t\tFalse Negative Rate (FNR): {100 * fnr:.2f}%')

    # ROC AUC
    unknown_class_idx = num_classes
    scores_unknown = openmax_probs_np[:, unknown_class_idx]

    fpr_roc, tpr_roc, _ = roc_curve(binary_labels, scores_unknown)
    roc_auc = auc(fpr_roc, tpr_roc)
    print(f'\n\t\tROC AUC: {roc_auc:.2f}')

    # Metrics
    eval_openmax = Evaluation(pred_openmax, labels, target_names)
    print('\n\t\tF1 Macro: {:.2f}%'.format(100 * eval_openmax.f1_macro))
    print('\t\tAccuracy: {:.2f}%'.format(100 * eval_openmax.accuracy))
    print('\t\tPrecision Macro: {:.2f}%'.format(100 * eval_openmax.precision_macro))
    print('\t\tRecall Macro: {:.2f}%'.format(100 * eval_openmax.recall_macro))
    print('\t\tClassification Report:\n', eval_openmax.class_report)

    return labels, scores_unknown
