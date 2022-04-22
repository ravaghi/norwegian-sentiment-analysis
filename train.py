import data.norec.dataloader as dataloader
import utils.preprocessing as preprocessing
from utils.visualization import plot_histories

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout, Bidirectional
from keras.optimizers import adam_v2
from collections import Counter
import numpy as np
import pandas as pd
import math
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = {
    "b-imbalanced": dataloader.load_full_dataset(binary=True),
    "b-balanced": dataloader.load_balanced_dataset(binary=True),
    "mc-imbalanced": dataloader.load_full_dataset(binary=False),
    "mc-balanced": dataloader.load_balanced_dataset(binary=False)
}


def get_vocab_size(texts):
    num_words = Counter()
    for text in texts:
        sentences = text.split(" ")
        for word in sentences:
            num_words[word] += 1

    num_words = len(num_words) / 20
    num_words = math.ceil(num_words / 1000) * 1000

    return num_words


def get_data(dataset):
    train = dataset["train"]
    val = dataset["dev"]
    test = dataset["test"]

    # Cleaning values in the text column
    train = preprocessing.clean_text(train, "text")
    val = preprocessing.clean_text(val, "text")
    test = preprocessing.clean_text(test, "text")

    # Combining data for later use
    combined_data = pd.concat([train, val, test]).reset_index(drop=True)

    # Separating texts and labels
    X_train, y_train = train["text"], train["label"]
    X_val, y_val = val["text"], val["label"]
    X_test, y_test = test["text"], test["label"]

    # Fitting a tokenizer to text from the combined data
    num_words = get_vocab_size(combined_data["text"])
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_data["text"].tolist())

    # Converting texts to sequences
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Deciding embedding vector length
    maxlen = (int(np.ceil(np.mean([len(text.split()) for text in combined_data.text]))))
    # Padding sequences with zeros until they reach a certain length
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    # Number of unique classes in the dataset
    num_classes = len(np.unique(y_train))
    # One-hot encoding of labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "tokenizer": tokenizer,
        "maxlen": maxlen,
        "num_classes": num_classes,
        "num_words": num_words
    }


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
    model.add(SpatialDropout1D(dropout))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units)))
    else:
        model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
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
    model.add(SpatialDropout1D(dropout))
    if bilstm:
        model.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))))
    else:
        model.add(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


if __name__ == '__main__':
    EPOCHS = 20
    BATCH_SIZE = 32
    EMBEDDING_DIM = 100
    LSTM_UNITS = 32
    L1_FACTOR = 0.001
    L2_FACTOR = 0.1
    LEARNING_RATE = 1e-5
    DROPOUT = 0.5
    optimizer = adam_v2.Adam(learning_rate=LEARNING_RATE)

    histories = {}
    results = []

    for dataset_name, dataset in datasets.items():
        processed_data = get_data(dataset)
        X_train = processed_data["X_train"]
        X_val = processed_data["X_val"]
        X_test = processed_data["X_test"]
        y_train = processed_data["y_train"]
        y_val = processed_data["y_val"]
        y_test = processed_data["y_test"]
        tokenizer = processed_data["tokenizer"]
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
