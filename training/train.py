import data.norec_sentence.dataloader as dataloader
from data.dataloader import load_data
from utils.visualization import plot_histories
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from keras.optimizers import adam_v2
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = {
    "imbalanced": dataloader.load_full_dataset(),
    "balanced": dataloader.load_balanced_dataset()
}


def load_best_hps():
    with open("best_hps.json", "r") as f:
        best_hps = json.load(f)

    return best_hps.get("values")


def baseline_model(embedding_dim, num_words, maxlen, num_classes, lstm_units,
                   bilstm, optimizer):
    model = Sequential(name="baseline")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units)))
    else:
        model.add(LSTM(lstm_units))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_1(embedding_dim, num_words, maxlen, num_classes, lstm_units,
            bilstm, optimizer, dropout):
    model = Sequential(name=f"baseline-dropout")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units)))
    else:
        model.add(LSTM(lstm_units, dropout=dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_2(embedding_dim, num_words, maxlen, num_classes, lstm_units,
            bilstm, optimizer, l1_factor, l2_factor):
    model = Sequential(name=f"baseline-regularization")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))))
    else:
        model.add(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_3(embedding_dim, num_words, maxlen, num_classes, lstm_units,
            bilstm, optimizer, dropout, l1_factor, l2_factor):
    model = Sequential(name=f"baseline-dropout-regularization")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units, dropout=dropout,
                                     kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))))
    else:
        model.add(LSTM(lstm_units, dropout=dropout, kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


if __name__ == '__main__':
    best_hps = load_best_hps()

    EPOCHS = 20
    BATCH_SIZE = 32
    EMBEDDING_DIM = best_hps["embedding_dim"]
    LSTM_UNITS = best_hps["lstm_units"]
    L1_FACTOR = best_hps["l1_regularizer"]
    L2_FACTOR = best_hps["l2_regularizer"]
    LEARNING_RATE = best_hps["learning_rate"]
    DROPOUT = best_hps["dropout"]
    optimizer = adam_v2.Adam(learning_rate=LEARNING_RATE)

    histories = {}
    results = []

    for dataset_name, dataset in datasets.items():
        processed_data = load_data(dataset)
        X_train = processed_data["X_train"]
        X_val = processed_data["X_val"]
        X_test = processed_data["X_test"]
        y_train = processed_data["y_train"]
        y_val = processed_data["y_val"]
        y_test = processed_data["y_test"]
        maxlen = processed_data["maxlen"]
        num_classes = processed_data["num_classes"]
        num_words = processed_data["num_words"]

        for bilstm in [True, False]:
            models = [
                baseline_model(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                               bilstm=bilstm, lstm_units=LSTM_UNITS, optimizer=optimizer),
                model_1(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                        bilstm=bilstm, lstm_units=LSTM_UNITS, optimizer=optimizer, dropout=DROPOUT),
                model_2(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                        bilstm=bilstm, lstm_units=LSTM_UNITS, optimizer=optimizer, l1_factor=L1_FACTOR,
                        l2_factor=L2_FACTOR),
                model_3(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                        bilstm=bilstm, lstm_units=LSTM_UNITS, optimizer=optimizer, dropout=DROPOUT, l1_factor=L1_FACTOR,
                        l2_factor=L2_FACTOR)
            ]

            for model in models:
                model_name = dataset_name + "-" + model.name
                lstm_type = "bilstm" if bilstm else "lstm"

                logdir = os.path.join(BASE_DIR, f"logs/{lstm_type}/{model_name}")
                # Early stopping to prevent overtraining when the model starts to overfit
                # And logging the history of the model to tensorboard
                callbacks = [EarlyStopping(monitor="val_accuracy", patience=5),
                             EarlyStopping(monitor="val_loss", patience=5),
                             TensorBoard(log_dir=logdir)]

                print(f"\n--------------- Training model: {lstm_type}-{model_name} ---------------")
                history = model.fit(X_train, y_train,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=(X_val, y_val),
                                    verbose=1,
                                    callbacks=callbacks)
                print(f"----------- Finished training model: {lstm_type}-{model_name} -----------\n")

                val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)

                model.save(os.path.join(BASE_DIR, f"models/{lstm_type}/{model_name}-{val_acc}.h5"))

                if lstm_type in histories.keys():
                    if dataset_name in histories[lstm_type].keys():
                        histories[lstm_type][dataset_name][model.name] = history
                    else:
                        histories[lstm_type][dataset_name] = {model.name: history}
                else:
                    histories[lstm_type] = {dataset_name: {model.name: history}}

                results.append(
                    {
                        "dataset": dataset_name,
                        "model": model.name,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lstm_type": lstm_type,
                        "hyperparameters": {
                            "embedding_dim": EMBEDDING_DIM,
                            "num_words": num_words,
                            "maxlen": maxlen,
                            "num_classes": num_classes,
                            "lstm_units": LSTM_UNITS,
                            "dropout": DROPOUT,
                            "l1_factor": L1_FACTOR,
                            "l2_factor": L2_FACTOR,
                            "optimizer_learning_rate": LEARNING_RATE
                        }
                    }
                )

    # Save parameters and results to a json file
    with open(os.path.join(BASE_DIR, "models/results.json"), "w") as f:
        json.dump(results, f)

    # Plot histories
    for lstm_type, lstm_histories in histories.items():
        for dataset_name, history in histories[lstm_type].items():
            plot_histories(history, f"{lstm_type}/{dataset_name}")
