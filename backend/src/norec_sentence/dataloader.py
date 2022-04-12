import json
import pandas as pd


def load_full_dataset() -> pd.DataFrame:
    data = []
    for name in ["train", "test", "dev"]:
        with open(f"3class/{name}.json", encoding="utf-8") as file:
            data = data + json.load(file)
    # Convert to pandas dataframe
    data = pd.DataFrame(data)
    # Convert labels to numerical values, and add them to the dataframe
    data["label"] = data["label"].replace({"Positive": 2, "Neutral": 1, "Negative": 0})
    return data


def load_balanced_dataset() -> pd.DataFrame:
    data = load_full_dataset()

    postives_df = data[data["label"] == 2]
    neutrals_df = data[data["label"] == 1]
    negatives_df = data[data["label"] == 0]

    smallest_size = min(postives_df.shape[0], neutrals_df.shape[0], negatives_df.shape[0])

    postives_df = postives_df.sample(n=smallest_size)
    neutrals_df = neutrals_df.sample(n=smallest_size)
    negatives_df = negatives_df.sample(n=smallest_size)

    return pd.concat([postives_df, neutrals_df, negatives_df])


def load_binary_dataset() -> pd.DataFrame:
    data = []
    for name in ["train", "test", "dev"]:
        with open(f"binary/{name}.json", encoding="utf-8") as file:
            data = data + json.load(file)
    # Convert to pandas dataframe
    data = pd.DataFrame(data)
    # Convert labels to numerical values, and add them to the dataframe
    data["label"] = data["label"].replace({"Positive": 1, "Negative": 0})

    return data


def load_balanced_binary_dataset() -> pd.DataFrame:
    data = load_binary_dataset()

    postives_df = data[data["label"] == 1]
    negatives_df = data[data["label"] == 0]

    smallest_size = min(postives_df.shape[0], negatives_df.shape[0])

    postives_df = postives_df.sample(n=smallest_size)
    negatives_df = negatives_df.sample(n=smallest_size)

    return pd.concat([postives_df, negatives_df])
