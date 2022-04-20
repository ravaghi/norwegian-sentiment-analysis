import data.norec.dataloader as dataloader
import utils.preprocessing as preprocessing
from utils.visualization import plot_histories

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout
from keras.optimizers import adam_v2
from collections import Counter
import numpy as np
import pandas as pd
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

    # Cleaning values in the text column
    train = preprocessing.clean_text(train, "text")
    val = preprocessing.clean_text(val, "text")

    # Combining data for later use
    combined_data = pd.concat([train, val]).reset_index(drop=True)

    # Separating texts and labels
    X_train, y_train = train["text"], train["label"]
    X_val, y_val = val["text"], val["label"]

    # Fitting a tokenizer to text from the combined data
    num_words = get_vocab_size(combined_data["text"])
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_data["text"].tolist())

    # Converting texts to sequences
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    # Deciding embedding vector length
    maxlen = (int(np.ceil(np.mean([len(text.split()) for text in combined_data.text]))))
    # Padding sequences with zeros until they reach a certain length
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)

    # Number of unique classes in the dataset
    num_classes = len(np.unique(y_train))
    # One-hot encoding of labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "maxlen": maxlen,
        "num_classes": num_classes,
        "num_words": num_words
    }


def dropout_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words,
                            lstm_units, embedding_dim, epochs, batch_size):
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    histories = {}
    for dropout in dropouts:
        print(f"Training model with dropout {dropout}")
        model = Sequential()
        model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
        model.add(SpatialDropout1D(dropout))
        model.add(LSTM(lstm_units))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=1)

        histories[f"dropout-{dropout}"] = history
    plot_histories(histories, "optimal_dropout")


def learning_rate_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words, lstm_units,
                                  embedding_dim, epochs, batch_size):
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    histories = {}
    for learning_rate in learning_rates:
        print(f"Training model with learning rate {learning_rate}")
        model = Sequential()
        model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
        model.add(LSTM(lstm_units))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer=adam_v2.Adam(learning_rate=learning_rate),
                      metrics=["accuracy"])

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=1)

        histories[f"lr-{learning_rate}"] = history
    plot_histories(histories, "optimal_learning_rate")


def l1_regularizer_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words, lstm_units,
                                   embedding_dim, epochs, batch_size):
    reg_factors = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    histories = {}
    for reg_factor in reg_factors:
        print(f"Training model with l1 regularization factor {reg_factor}")
        model = Sequential()
        model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
        model.add(LSTM(lstm_units, kernel_regularizer=l1(reg_factor)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=1)

        histories[f"l1-{reg_factor}"] = history
    plot_histories(histories, "optimal_l1")


def l2_regularizer_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words, lstm_units,
                                   embedding_dim, epochs, batch_size):
    reg_factors = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    histories = {}
    for reg_factor in reg_factors:
        print(f"Training model with l2 regularization factor {reg_factor}")
        model = Sequential()
        model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
        model.add(LSTM(lstm_units, kernel_regularizer=l2(reg_factor)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=1)

        histories[f"l2-{reg_factor}"] = history
    plot_histories(histories, "optimal_l2")


if __name__ == '__main__':
    EPOCHS = 15
    BATCH_SIZE = 32

    EMBEDDING_DIM = 100
    LSTM_UNITS = 32

    dataset = dataloader.load_full_dataset()
    processed_data = get_data(dataset)
    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    y_train = processed_data["y_train"]
    y_val = processed_data["y_val"]
    maxlen = processed_data["maxlen"]
    num_classes = processed_data["num_classes"]
    num_words = processed_data["num_words"]

    dropout_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words,
                            LSTM_UNITS, EMBEDDING_DIM, EPOCHS, BATCH_SIZE)
    learning_rate_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words,
                                  LSTM_UNITS, EMBEDDING_DIM, EPOCHS, BATCH_SIZE)
    l1_regularizer_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words,
                                   LSTM_UNITS, EMBEDDING_DIM, EPOCHS, BATCH_SIZE)
    l2_regularizer_optimizer_model(X_train, X_val, y_train, y_val, maxlen, num_classes, num_words,
                                   LSTM_UNITS, EMBEDDING_DIM, EPOCHS, BATCH_SIZE)
