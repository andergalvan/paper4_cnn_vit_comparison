import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def save_train_val_metrics(epochs, train_losses, train_accuracies, val_losses, val_accuracies, path):
    
    """ Save training and validation losses anda accuracies. """

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Losses (Training and Validation)
    min_train_loss_epoch = train_losses.index(min(train_losses)) + 1
    min_train_loss = min(train_losses)
    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    min_val_loss = min(val_losses)

    ax[0].plot(range(1, epochs + 1), train_losses, label=f'Training Loss (min: {min_train_loss:.4f} @epoch {min_train_loss_epoch})', color='green')
    ax[0].plot(range(1, epochs + 1), val_losses, label=f'Validation Loss (min: {min_val_loss:.4f} @epoch {min_val_loss_epoch})', color='orange')
    ax[0].scatter(min_train_loss_epoch, min_train_loss, color='red')
    ax[0].scatter(min_val_loss_epoch, min_val_loss, color='purple')
    ax[0].set_title('Losses')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # Subplot 2: Accuracies (Training and Validation)
    max_train_accuracy_epoch = train_accuracies.index(max(train_accuracies)) + 1
    max_train_accuracy = max(train_accuracies)
    max_val_accuracy_epoch = val_accuracies.index(max(val_accuracies)) + 1
    max_val_accuracy = max(val_accuracies)

    ax[1].plot(range(1, epochs + 1), train_accuracies, label=f'Training Accuracy (max: {max_train_accuracy:.4f} @epoch {max_train_accuracy_epoch})', color='blue')
    ax[1].plot(range(1, epochs + 1), val_accuracies, label=f'Validation Accuracy (max: {max_val_accuracy:.4f} @epoch {max_val_accuracy_epoch})', color='cyan')
    ax[1].scatter(max_train_accuracy_epoch, max_train_accuracy, color='red')
    ax[1].scatter(max_val_accuracy_epoch, max_val_accuracy, color='purple')
    ax[1].set_title('Accuracies')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def save_confusion_matrix(labels, preds, class_names, path):

    """ Save confusion matrix. """

    cm = confusion_matrix(labels, preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(50, 50))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)

    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(labels, predictions,unknown_class_labels, save_path):
    
    """ Plot ROC Curve. """

    labels = np.array(labels)
    predictions = np.array(predictions)
    
    binary_labels = (labels == unknown_class_labels).astype(int)

    fpr, tpr, thresholds = roc_curve(binary_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(12, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    
    # Add labels and title
    plt.xlabel('False Positive Rate (FPR)', fontsize=18)
    plt.ylabel('True Positive Rate (TPR)', fontsize=18)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    
    # Customize ticks and legend
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    # Add grid and save the plot
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()