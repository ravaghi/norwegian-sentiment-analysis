import sena.data.norec.dataloader as dataloader
import sena.utils.preprocessing as preprocessing

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout
from keras.optimizers import adam_v2
from collections import Counter
import numpy as np
import pandas as pd
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = {
    "bnf": dataloader.load_full_dataset(binary=True),
    "bnb": dataloader.load_balanced_dataset(binary=True),
    "mnf": dataloader.load_full_dataset(binary=False),
    "mnb": dataloader.load_balanced_dataset(binary=False)
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


def baseline_model(embedding_dim, num_words, maxlen, num_classes, lstm_units, optimizer, dropout=None,
                   regularizer=None):
    model = Sequential(name="baseline")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    model.add(LSTM(lstm_units))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_1(embedding_dim, num_words, maxlen, num_classes, lstm_units, optimizer, dropout=None, regularizer=None):
    model = Sequential(name=f"baseline-dropout")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_2(embedding_dim, num_words, maxlen, num_classes, lstm_units, optimizer, dropout=None, regularizer=None):
    model = Sequential(name=f"baseline-regularization")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    model.add(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=regularizer, l2=regularizer)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def model_3(embedding_dim, num_words, maxlen, num_classes, lstm_units, optimizer, dropout=None, regularizer=None):
    model = Sequential(name=f"baseline-dropout-regularization")
    model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(lstm_units, kernel_regularizer=l1_l2(l1=regularizer, l2=regularizer)))
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
    LSTM_UNITS = 64

    REGULARIZER = 0.0003
    LEARNING_RATE = 0.001
    DROPOUT = 0.4

    optimizer = adam_v2.Adam(learning_rate=LEARNING_RATE)

    histories = {}

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

        models = [
            baseline_model(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                           lstm_units=LSTM_UNITS, optimizer=optimizer),
            model_1(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                    lstm_units=LSTM_UNITS, optimizer=optimizer, dropout=DROPOUT),
            model_2(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                    lstm_units=LSTM_UNITS, optimizer=optimizer, regularizer=REGULARIZER),
            model_3(embedding_dim=EMBEDDING_DIM, num_words=num_words, maxlen=maxlen, num_classes=num_classes,
                    lstm_units=LSTM_UNITS, optimizer=optimizer, dropout=DROPOUT, regularizer=REGULARIZER)
        ]

        for model in models:
            model_name = dataset_name + "-" + model.name

            # Early stopping to prevent overtraining when the model starts to overfit
            # And logging the history of the model to tensorboard
            callbacks = [EarlyStopping(monitor="val_accuracy", patience=3),
                         EarlyStopping(monitor="val_loss", patience=3),
                         TensorBoard(log_dir=os.path.join(BASE_DIR, f"logs/{model_name}"))]

            print(f"\n--------------- Training model: {model_name} ---------------")

            history = model.fit(X_train, y_train,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(X_val, y_val),
                                verbose=1,
                                callbacks=callbacks)

            val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)
            print("Validation loss:", val_loss)
            print("Validation accuracy:", val_acc)

            if val_acc > 0.8:
                print(f"Saving model {model_name} ...")
                model.save(os.path.join(BASE_DIR, f"models/{model_name}.h5"))

            histories[dataset_name][model.name] = history

            print(f"----------- Finished training model: {model_name} -----------\n")
