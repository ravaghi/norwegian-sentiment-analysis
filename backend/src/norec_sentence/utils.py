import pandas as pd


def clean_text(dataframe, column_name) -> pd.DataFrame:
    """ Clean text from dataframe.

    Args:
        dataframe: Dataframe to clean.
        column_name: Column name to clean.

    Returns: Cleaned dataframe.

    """
    # Remove non-alphabetical characters
    # dataframe[column_name] = dataframe[column_name].str.replace(r'[^a-zA-Z]', ' ')
    dataframe[column_name] = dataframe[column_name].str.lower()
    dataframe[column_name] = dataframe[column_name].str.strip()
    return dataframe
