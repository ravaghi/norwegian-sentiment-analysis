import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import pickle
from collections import Counter

from keras import regularizers
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import data.norec.dataloader as norec_dataloader
import data.norec_sentence.dataloader as norec_sentence_dataloader
import utils.preprocessing as preprocessing
from utils.visualization import plot_histories

# NoReC data
norec_multiclass_dataset = norec_dataloader.load_full_dataset()
norec_balanced_multiclass_dataset = norec_dataloader.load_balanced_dataset()
norec_binary_dataset = norec_dataloader.load_binary_dataset()
norec_balanced_binary_dataset = norec_dataloader.load_balanced_binary_dataset()

# NoReC sentence data
norec_sentence_multiclass_dataset = norec_sentence_dataloader.load_full_dataset()
norec_sentence_balanced_multiclass_dataset = norec_sentence_dataloader.load_balanced_dataset()
norec_sentence_binary_dataset = norec_sentence_dataloader.load_binary_dataset()
norec_sentence_balanced_binary_dataset = norec_sentence_dataloader.load_balanced_binary_dataset()

datasets = {
    "binary": [norec_binary_dataset,
               norec_balanced_binary_dataset,
               norec_sentence_binary_dataset,
               norec_sentence_balanced_binary_dataset],
    "multiclass": [norec_multiclass_dataset,
                   norec_balanced_multiclass_dataset,
                   norec_sentence_multiclass_dataset,
                   norec_sentence_balanced_multiclass_dataset]
}


def get_vocab_size(text):
    num_words = Counter()
    for t in text:
        for word in t.split(" "):
            num_words[word] += 1
    num_words = len(num_words) / 20
    return math.ceil(num_words / 1000) * 1000


def prepare_data(dataset, dataset_name):
    print(f"Preparing {dataset_name} ...")
    train = dataset["train"]
    val = dataset["dev"]

    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)

    train = preprocessing.clean_text(train, "text")
    val = preprocessing.clean_text(val, "text")

    combined_data = pd.concat([train, val]).reset_index(drop=True)

    X_train, y_train = train["text"], train["label"]
    X_val, y_val = val["text"], val["label"]

    num_words = get_vocab_size(combined_data["text"])
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_data["text"].tolist())
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    maxlen = (int(np.ceil(np.mean([len(text.split()) for text in combined_data.text]))))
    X_train = pad_sequences(X_train, maxlen=maxlen, padding="post", truncating="post")
    X_val = pad_sequences(X_val, maxlen=maxlen, padding="post", truncating="post")

    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "maxlen": maxlen,
        "num_classes": num_classes,
        "num_words": num_words
    }


def model_1(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_1")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_2(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_2")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_3(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_3")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.6, recurrent_dropout=0.6))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_4(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_4")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, recurrent_regularizer=regularizers.l2(0.01)))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_5(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_5")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, recurrent_regularizer=regularizers.l2(0.01),
                     return_sequences=True))
    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, recurrent_regularizer=regularizers.l2(0.01)))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_6(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_6")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, return_sequences=True))
    model_x.add(LSTM(64))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_7(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_7")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, return_sequences=True))
    model_x.add(LSTM(64))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_8(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_8")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(Bidirectional(LSTM(64, return_sequences=True)))
    model_x.add(Bidirectional(LSTM(64)))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_9(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_9")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(Bidirectional(LSTM(64, return_sequences=True)))
    model_x.add(Bidirectional(LSTM(64)))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_10(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_10")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(Bidirectional(LSTM(64, return_sequences=True)))
    model_x.add(
        Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01))))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_11(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_11")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(Bidirectional(LSTM(64, return_sequences=True)))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_12(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_12")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def model_13(num_words, maxlen, num_classes):
    model_x = Sequential(name=f"model_13")

    model_x.add(Embedding(num_words, 100, input_length=maxlen))

    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model_x.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))
    model_x.add(Dense(64, activation="relu"))
    model_x.add(Dropout(0.3))

    model_x.add(Dense(num_classes, activation="softmax"))
    model_x.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_x


def get_models(num_words, maxlen, num_classes):
    return [
        model_1(num_words, maxlen, num_classes),
        model_2(num_words, maxlen, num_classes),
        model_3(num_words, maxlen, num_classes),
        model_4(num_words, maxlen, num_classes),
        model_5(num_words, maxlen, num_classes),
        model_6(num_words, maxlen, num_classes),
        model_7(num_words, maxlen, num_classes),
        model_8(num_words, maxlen, num_classes),
        model_9(num_words, maxlen, num_classes),
        model_10(num_words, maxlen, num_classes),
        model_11(num_words, maxlen, num_classes),
        model_12(num_words, maxlen, num_classes),
        model_13(num_words, maxlen, num_classes),
    ]


if __name__ == '__main__':
    epochs = 20
    batch_size = 32

    histories = {
        "binary": [],
        "multiclass": []
    }

    for dataset_type, datasets in datasets.items():
        for dataset in datasets:
            dataset_name = [name for name in globals() if globals()[name] is dataset][0]
            data = prepare_data(dataset, dataset_name)
            models = get_models(data["num_words"], data["maxlen"], data["num_classes"])
            for model in models:
                print(f"\n--------------- Training {model.name} on {dataset_name} ---------------")
                try:
                    history = model.fit(data["X_train"],
                                        data["y_train"],
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(data["X_val"], data["y_val"]),
                                        verbose=2)

                    histories[dataset_type].append({
                        "model": model.name,
                        "history": history,
                        "dataset": dataset_name
                    })
                except Exception as e:
                    continue
                print(f"---------- Finshed training {model.name} on {dataset_name} ----------\n")

    # Save histories into pickle
    with open("histories.pkl", "wb") as f:
        pickle.dump(histories, f)

    plot_histories(histories)
