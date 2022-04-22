import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_full_dataset(binary=False) -> dict:
    """Loads the full dataset.

    Args:
        binary: Whether to load the binary or multiclass dataset.

    Returns: A dictionary of dataframes containing datasets.

    """
    data = {}
    for name in ["train", "test", "dev"]:
        dataset_type = "binary" if binary else "multiclass"
        with open(os.path.join(BASE_DIR, f"{dataset_type}/{name}.json"), encoding="utf-8") as file:
            # Convert to pandas dataframe
            df = pd.DataFrame.from_dict(json.load(file))

            # Remove sentence ids
            df = df.drop(columns=["sent_id"])

        # Convert labels to numerical values
        if binary:
            df["label"] = df["label"].replace({"Negative": 0, "Positive": 1})
        else:
            df["label"] = df["label"].replace({"Negative": 0, "Neutral": 1, "Positive": 2})

        # Add the dataframe to the dictionary and shuffle
        data[name] = df.sample(frac=1).reset_index(drop=True)
    return data


def load_balanced_dataset(binary=False) -> dict:
    """Loads the balanced dataset.

    Args:
        binary: Whether to load the binary or multiclass dataset.

    Returns: A dictionary of dataframes containing balanced (undersampled) datasets.

    """
    data = load_full_dataset(binary=binary)

    balanced_data = {}
    for df_name, df in data.items():
        if binary:
            negatives_df = df[df["label"] == 0]
            postives_df = df[df["label"] == 1]

            # Undersample the minority class
            smallest_size = min(negatives_df.shape[0], postives_df.shape[0])

            # Resample the majority classes
            negatives_df = negatives_df.sample(n=smallest_size)
            postives_df = postives_df.sample(n=smallest_size)

            # Concatenate the results
            result = pd.concat([negatives_df, postives_df]).reset_index(drop=True)
        else:
            negatives_df = df[df["label"] == 0]
            neutrals_df = df[df["label"] == 1]
            postives_df = df[df["label"] == 2]

            # Undersample the minority class
            smallest_size = min(postives_df.shape[0], neutrals_df.shape[0], negatives_df.shape[0])

            # Resample the majority classes
            negatives_df = negatives_df.sample(n=smallest_size)
            neutrals_df = neutrals_df.sample(n=smallest_size)
            postives_df = postives_df.sample(n=smallest_size)

            # Concatenate the results and shuffle
            result = pd.concat([postives_df, neutrals_df, negatives_df]).sample(frac=1).reset_index(drop=True)

        balanced_data[df_name] = result

    return balanced_data
