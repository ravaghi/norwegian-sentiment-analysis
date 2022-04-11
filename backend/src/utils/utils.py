import pandas as pd
import json
import os
import string

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
    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    return dataframe


def remove_punctuation(dataframe, column_name) -> pd.DataFrame:
    """Remove punctuations from dataframe.
        Args:
            dataframe: Dataframe to clean.
            column_name: Column name to clean.

        Returns: Cleaned dataframe.

        """
    dataframe[column_name] = dataframe[column_name].str.replace(r'[{}]'.format(string.punctuation), ' ', regex=True)
    return dataframe


def clean_text(dataframe, column_name) -> pd.DataFrame:
    """ Clean text from dataframe.

    Args:
        dataframe: Dataframe to clean.
        column_name: Column name to clean.

    Returns: Cleaned dataframe.

    """
    stopwords = load_stopwords()
    dataframe = remove_stopwords(dataframe, column_name, stopwords)
    dataframe = remove_punctuation(dataframe, column_name)
    dataframe[column_name] = dataframe[column_name].str.strip()
    return dataframe
