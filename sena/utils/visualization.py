import matplotlib.pyplot as plt


def plot_history(history):
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
    plt.show()


def plot_histories(histories):
    for dataset_type, model_histories in histories.items():
        plt.style.use("ggplot")
        plt.figure(figsize=(20, 10))
        for model_history in model_histories:
            model_name = model_history["model"]
            dataset_name = model_history["dataset"]

            acc = model_history["history"].history['val_accuracy']
            loss = model_history["history"].history['val_loss']

            label = f"{model_name}-{dataset_name}"

            plt.subplot(2, 2, 1)
            plt.plot(acc, label=label)
            plt.title(f'Validation Accuracy')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(loss, label=label)
            plt.title(f'Validation Loss')
            plt.legend()

        plt.savefig(dataset_type + "_training.png")
        plt.show()
