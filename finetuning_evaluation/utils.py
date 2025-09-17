import random
import numpy as np
import copy

import torch
import torch.nn as nn

from torchvision.models import resnet50, mobilenet_v3_large, inception_v3, vit_b_16, densenet201, efficientnet_b0

from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score, confusion_matrix

from evaluation import *
from openmax import *
from plot import *


def set_seed(seed):
    
    """ Set seed. """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    
    """ Fine-tune the model (training and validation) with loss, recall and precision. """

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_rvalues, train_pvalues = [], []
    val_rvalues, val_pvalues = [], []
    
    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []

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
            
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Precision and Recall
        try:
            p_value = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
            r_value = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
        except:
            p_value, r_value = 0.0, 0.0
        train_pvalues.append(p_value)
        train_rvalues.append(r_value)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Precision and Recall
        try:
            p_value = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
            r_value = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
        except:
            p_value, r_value = 0.0, 0.0
        val_pvalues.append(p_value)
        val_rvalues.append(r_value)

        print(f'\t\tEpoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Recall: {train_rvalues[-1]:.2f}%, Train Precision: {train_pvalues[-1]:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val Recall: {val_rvalues[-1]:.2f}%, Val Precision: {val_pvalues[-1]:.2f}%')

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = copy.deepcopy(model)
            print(f'\t\t\t✅ Best model updated: Val Acc = {best_val_acc:.2f}%')

    return {
        'best_model': best_model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'train_rvalues': train_rvalues,
        'train_pvalues': train_pvalues,
        'val_rvalues': val_rvalues,
        'val_pvalues': val_pvalues
    }


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

    test_f1 = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_report = classification_report(all_labels, all_preds, target_names=class_names_close)
    test_precision = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_recall = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0.0)
    test_accuracy = 100 * accuracy_score(all_labels, all_preds)

    print('\n\t\t\tF1 Macro: {:.2f}%'.format(test_f1))
    print('\t\t\tAccuracy: {:.2f}%'.format(test_accuracy))
    print('\t\t\tPrecision: {:.2f}%'.format(test_precision))
    print('\t\t\tRecall: {:.2f}%'.format(test_recall))
    print('\t\t\tClassification Report:\n', test_report)
    
    return {"f1_macro": test_f1}


def test_open_openmax(net, opentest_loader, device, weibull_model, num_classes, target_names, weibull_alpha, weibull_threshold, weibull_distance): 
    
    """ Evaluation on OSR scenario using OpenMax. """

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
    for score in scores:
        categories = list(range(0, num_classes))
        so, ss = openmax(weibull_model, categories, score, 0.5, weibull_alpha, weibull_distance)
        pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else num_classes)

    pred_openmax_np = np.array(pred_openmax)

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

    print(f'\n\t\t\tFalse Positive Rate (FPR): {100 * fpr:.2f}%')
    print(f'\t\t\tTrue Negative Rate (TNR): {100 * tnr:.2f}%')
    print(f'\t\t\tTrue Positive Rate (TPR): {100 * tpr:.2f}%')
    print(f'\t\t\tFalse Negative Rate (FNR): {100 * fnr:.2f}%')

    eval_openmax = Evaluation(pred_openmax, labels, target_names)
    print('\n\t\t\tF1 Macro: {:.2f}%'.format(100 * eval_openmax.f1_macro))
    print('\t\t\tAccuracy: {:.2f}%'.format(100 * eval_openmax.accuracy))
    print('\t\t\tPrecision Macro: {:.2f}%'.format(100 * eval_openmax.precision_macro))
    print('\t\t\tRecall Macro: {:.2f}%'.format(100 * eval_openmax.recall_macro))
    print('\t\t\tClassification Report:\n', eval_openmax.class_report)

    return {
        "labels": labels,
        "f1_macro": 100*eval_openmax.f1_macro,
        "fpr": 100*fpr,
        "tnr": 100*tnr,
        "tpr": 100*tpr,
        "fnr": 100*fnr
    }
    
    
def test_open_error_analysis(net, opentest_loader, device, weibull_model, num_classes, target_names, weibull_alpha, weibull_threshold, weibull_distance): 
    
    """ Function used for visual error analysis that return true labels and predicted labels using OpenMax. """

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

    # Convert labels and predictions to strings to match 'unknown' format
    labels_true_str = [target_names[l] if l < num_classes else 'unknown' for l in labels]
    labels_pred_str = [target_names[p] if p < num_classes else 'unknown' for p in pred_openmax_np]

    return labels_true_str, labels_pred_str


def test_open_softmax_thresholding(net, opentest_loader, device, weibull_model, num_classes, target_names, weibull_alpha, weibull_threshold, weibull_distance): 
    
    """ Evaluation on OSR scenario using SoftMax Thresholding. """

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

    # SoftMax Thresholding predictions
    pred_softmax_thresholding = []

    for score in scores:
        categories = list(range(0, num_classes))
        so, ss = openmax(weibull_model, categories, score, 0.5, weibull_alpha, weibull_distance)
        pred_softmax_thresholding.append(np.argmax(ss) if np.max(ss) >= weibull_threshold else num_classes)

    pred_softmax_thresholding_np = np.array(pred_softmax_thresholding)

    # Binary labels (1: unknown, 0: known)
    binary_labels = (labels == num_classes).astype(int)
    binary_preds = (pred_softmax_thresholding_np == num_classes).astype(int)

    # Confusion matrix
    cm_binary = confusion_matrix(binary_labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_binary[0, 0], cm_binary[0, 1], cm_binary[1, 0], cm_binary[1, 1]

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f'\n\t\t\tFalse Positive Rate (FPR): {100 * fpr:.2f}%')
    print(f'\t\t\tTrue Negative Rate (TNR): {100 * tnr:.2f}%')
    print(f'\t\t\tTrue Positive Rate (TPR): {100 * tpr:.2f}%')
    print(f'\t\t\tFalse Negative Rate (FNR): {100 * fnr:.2f}%')

    eval_softmax_thresholding = Evaluation(pred_softmax_thresholding, labels, target_names)
    print('\n\t\t\tF1 Macro: {:.2f}%'.format(100 * eval_softmax_thresholding.f1_macro))
    print('\t\t\tAccuracy: {:.2f}%'.format(100 * eval_softmax_thresholding.accuracy))
    print('\t\t\tPrecision Macro: {:.2f}%'.format(100 * eval_softmax_thresholding.precision_macro))
    print('\t\t\tRecall Macro: {:.2f}%'.format(100 * eval_softmax_thresholding.recall_macro))
    print('\t\t\tClassification Report:\n', eval_softmax_thresholding.class_report)

    return {
        "labels": labels,
        "f1_macro": 100*eval_softmax_thresholding.f1_macro,
        "fpr": 100*fpr,
        "tnr": 100*tnr,
        "tpr": 100*tpr,
        "fnr": 100*fnr
    }