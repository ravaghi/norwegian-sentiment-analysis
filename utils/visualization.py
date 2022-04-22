import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_history(history, plot_name):
    plt.style.use("ggplot")
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'training/plots')
    plt.savefig(os.path.join(plot_path, f"{plot_name}.pdf"), bbox_inches='tight')


def plot_histories(histories, plot_name):
    plt.style.use("ggplot")
    plt.figure(figsize=(20, 10))
    for name, history in histories.items():
        acc = history.history['val_accuracy']
        loss = history.history['val_loss']

        plt.subplot(2, 2, 1)
        plt.plot(acc, label=name)
        plt.title('Validation Accuracy')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(loss, label=name)
        plt.title('Validation Loss')
        plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'training/plots')
    plt.savefig(os.path.join(plot_path, f"{plot_name}.pdf"), bbox_inches='tight')
