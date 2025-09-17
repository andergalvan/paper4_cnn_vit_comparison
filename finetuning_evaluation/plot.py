import matplotlib.pyplot as plt


def save_train_val_metrics(epochs, train_losses, train_precisions, train_recalls, val_losses, val_precisions, val_recalls, path):
    
    """ Save training and validation metrics: loss, precision and recall. """
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Loss
    ax[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', color='green')
    ax[0].plot(range(1, epochs + 1), val_losses, label='Val Loss', color='red')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].legend()

    # Precision and Recall
    ax[1].plot(range(1, epochs + 1), train_precisions, label='Train Precision', color='blue')
    ax[1].plot(range(1, epochs + 1), val_precisions, label='Val Precision', color='cyan')
    ax[1].plot(range(1, epochs + 1), train_recalls, label='Train Recall', color='orange')
    ax[1].plot(range(1, epochs + 1), val_recalls, label='Val Recall', color='magenta')
    ax[1].set_title('Precision & Recall')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Score')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)