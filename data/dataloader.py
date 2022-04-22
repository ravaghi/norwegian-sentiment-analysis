import utils.preprocessing as preprocessing

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from collections import Counter
import numpy as np
import pandas as pd
import math


def get_vocab_size(texts):
    """Returns an approximation of the number of unique words in the dataset.
    Args:
        texts: A list of texts

    Returns: An integer representing the number of unique words in the dataset.

    """
    num_words = Counter()
    for text in texts:
        sentences = text.split(" ")
        for word in sentences:
            num_words[word] += 1

    # Shorten the number of the words to improve training
    num_words = len(num_words) / 15
    num_words = math.ceil(num_words / 1000) * 1000

    return num_words


def load_data(dataset):
    """Loads the dataset.
    Args:
        dataset: A string representing the dataset to load.

    Returns: A dictionary containing the loaded dataset and metadata.

    """
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
        "maxlen": maxlen,
        "num_classes": num_classes,
        "num_words": num_words
    }
