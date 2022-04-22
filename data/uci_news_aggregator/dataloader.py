import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_full_dataset() -> dict:
    """Loads the full dataset.

    Returns: A dictionary of dataframes containing datasets.

    """

    # Read csv file into dataframe
    df = pd.read_csv(os.path.join(BASE_DIR, 'uci-news-aggregator.csv'), usecols=['TITLE', 'CATEGORY'])

    # Rename columns
    df = df.rename(columns={"TITLE": "text", "CATEGORY": "label"})

    # Convert sentiment to numerical values
    df["label"] = df["label"].replace({"e": 0, "b": 1, "t": 2, "m": 3})

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    df_size = df.shape[0]

    # Split the dataframe into training, validation and test sets
    return {
        "train": df[:int(df_size * 0.8)],
        "dev": df[int(df_size * 0.8):int(df_size * 0.9)],
        "test": df[int(df_size * 0.9):]
    }
