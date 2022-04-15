import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metadata() -> dict:
    """Loads the metadata of the dataset.

    Returns: A dictionary containing the metadata.

    """
    with open(os.path.join(BASE_DIR, "data/metadata.json"), encoding="utf-8") as file:
        return json.load(file)


'''def load_full_dataset() -> dict:
    """Loads the full dataset.

    Returns: A dictionary of dataframes containing the multiclass datasets.

    """
    data = {}
    for name in ["train", "test", "dev"]:
        with open(os.path.join(BASE_DIR, f"{name}.json"), encoding="utf-8") as file:
            # Convert to pandas dataframe
            df = pd.DataFrame(json.load(file))
            # Convert labels to numerical values
            df["label"] = df["label"].replace({"Positive": 2, "Neutral": 1, "Negative": 0})
            data[name] = df
    return data


def load_balanced_dataset() -> dict:
    """Loads the balanced dataset.

    Returns: A dictionary of dataframes containing the multiclass datasets.

    """
    data = load_full_dataset()

    balanced_data = {}
    for df_name, df in data.items():
        postives_df = df[df["label"] == 2]
        neutrals_df = df[df["label"] == 1]
        negatives_df = df[df["label"] == 0]

        smallest_size = min(postives_df.shape[0], neutrals_df.shape[0], negatives_df.shape[0])

        postives_df = postives_df.sample(n=smallest_size)
        neutrals_df = neutrals_df.sample(n=smallest_size)
        negatives_df = negatives_df.sample(n=smallest_size)

        result = pd.concat([postives_df, neutrals_df, negatives_df])
        balanced_data[df_name] = result

    return balanced_data


def load_binary_dataset() -> dict:
    """Loads the binary dataset.

    Returns: A dictionary of dataframes containing the binary dataset.

    """
    data = {}
    for name in ["train", "test", "dev"]:
        with open(os.path.join(BASE_DIR, f"binary/{name}.json"), encoding="utf-8") as file:
            # Convert to pandas dataframe
            df = pd.DataFrame(json.load(file))
            # Convert labels to numerical values
            df["label"] = df["label"].replace({"Positive": 1, "Negative": 0})
            data[name] = df
    return data


def load_balanced_binary_dataset() -> dict:
    """Loads the binary dataset.

    Returns: A dictionary of dataframes containing the balanced binary datasets.

    """
    data = load_binary_dataset()

    balanced_data = {}
    for df_name, df in data.items():
        postives_df = df[df["label"] == 1]
        negatives_df = df[df["label"] == 0]

        smallest_size = min(postives_df.shape[0], negatives_df.shape[0])

        postives_df = postives_df.sample(n=smallest_size)
        negatives_df = negatives_df.sample(n=smallest_size)

        result = pd.concat([postives_df, negatives_df])
        balanced_data[df_name] = result

    return balanced_data'''

if __name__ == '__main__':
    print(load_metadata())
