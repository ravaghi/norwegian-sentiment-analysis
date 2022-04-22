from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch

import data.norec.dataloader as dataloader
import utils.preprocessing as preprocessing

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.optimizers import adam_v2
from collections import Counter
import numpy as np
import pandas as pd
import math
import os
import pickle

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


def build_model(hp):
    model = Sequential()

    model.add(
        Embedding(input_dim=num_words,
                  output_dim=hp.Int(
                      "embedding_dim",
                      min_value=10,
                      max_value=100,
                      step=10
                  ),
                  input_length=maxlen
                  )
    )
    model.add(
        SpatialDropout1D(
            hp.Float(
                "spatial_dropout",
                min_value=0.1,
                max_value=0.5,
                step=0.05
            )
        )
    )
    model.add(
        LSTM(
            units=hp.Int(
                "lstm_units",
                min_value=16,
                max_value=128,
                step=16
            ),
            dropout=hp.Float(
                "dropout",
                min_value=0.1,
                max_value=0.5,
                step=0.05
            ),
            recurrent_dropout=hp.Float(
                "recurrent_dropout",
                min_value=0.1,
                max_value=0.5,
                step=0.05
            ),
            kernel_regularizer=l1_l2(
                l1=hp.Choice(
                    "l1_regularizer",
                    values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
                ),
                l2=hp.Choice(
                    "l2_regularizer",
                    values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
                )
            ),
            recurrent_regularizer=l1_l2(
                l1=hp.Choice(
                    "recurrent_l1_regularizer",
                    values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
                ),
                l2=hp.Choice(
                    "recurrent_l2_regularizer",
                    values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
                )
            )
        )
    )
    model.add(
        Dense(
            units=num_classes,
            activation='softmax'
        )
    )
    model.compile(
        optimizer=adam_v2.Adam(
            hp.Float(
                'learning_rate',
                min_value=1e-4,
                max_value=1e-2,
                sampling='LOG',
                default=1e-3
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == '__main__':
    EPOCHS = 20
    BATCH_SIZE = 32

    dataset = dataloader.load_full_dataset()
    processed_data = get_data(dataset)
    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    y_train = processed_data["y_train"]
    y_val = processed_data["y_val"]
    maxlen = processed_data["maxlen"]
    num_classes = processed_data["num_classes"]
    num_words = processed_data["num_words"]

    callbacks = [EarlyStopping(monitor="val_accuracy", patience=5),
                 EarlyStopping(monitor="val_loss", patience=5)]

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=100,
        executions_per_trial=1,
        directory=BASE_DIR,
        project_name="sena",
    )

    print(tuner.search_space_summary())

    tuner.search(X_train,
                 y_train,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 validation_data=(X_val, y_val),
                 callbacks=callbacks)

    with open(f"tuner_results.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner = pickle.load(open("tuner_results.pkl", "rb"))
    print(tuner.get_best_hyperparameters()[0].values)
    print(tuner.results_summary())
    print(tuner.get_best_models()[0].summary())
