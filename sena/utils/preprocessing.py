import string
import pandas as pd
import json
import os
from nltk.corpus import stopwords as english_stopwords

ENGLISH_STOPWORDS = set(english_stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_stopwords() -> list:
    """Load stopwords from file.

    Returns: List of stopwords.

    """
    with open(os.path.join(BASE_DIR, "no_stopwords.json"), encoding="utf-8") as file:
        stopwords = json.load(file)

    return stopwords


def remove_stopwords(dataframe, column_name, stopwords) -> pd.DataFrame:
    """Remove stopwords from dataframe.

    Args:
        dataframe: Dataframe to clean.
        column_name: Column name to clean.
        stopwords: List of stopwords.

    Returns: Cleaned dataframe.

    """
    dataframe_copy = dataframe.copy()
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(lambda x: " ".join(
        [word for word in x.split() if
         word not in stopwords and word not in ENGLISH_STOPWORDS and word != "" and word.isalpha() and "\n" not in word]))
    return dataframe_copy


def remove_punctuation(dataframe, column_name) -> pd.DataFrame:
    """Remove punctuations from dataframe.

        Args:
            dataframe: Dataframe to clean.
            column_name: Column name to clean.

    Returns: Cleaned dataframe.

    """
    dataframe_copy = dataframe.copy()
    dataframe_copy[column_name] = dataframe_copy[column_name].str.translate(
        str.maketrans('', '', string.punctuation + string.digits))
    return dataframe_copy


def convert_to_lowercase(dataframe, column_name) -> pd.DataFrame:
    """Convert text to lowercase.

        Args:
            dataframe: Dataframe to clean.
            column_name: Column name to clean.

    Returns: Cleaned dataframe.

    """
    dataframe_copy = dataframe.copy()
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(lambda x: x.lower())
    return dataframe_copy


def clean_text(dataframe, column_name) -> pd.DataFrame:
    """ Clean text from dataframe.

        Args:
            dataframe: Dataframe to clean.
            column_name: Column name to clean.

    Returns: Cleaned dataframe.

    """
    stopwords = load_stopwords()
    dataframe = convert_to_lowercase(dataframe, column_name)
    dataframe = remove_punctuation(dataframe, column_name)
    dataframe = remove_stopwords(dataframe, column_name, stopwords)
    return dataframe
