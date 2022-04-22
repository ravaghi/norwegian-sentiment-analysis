import json
import pandas as pd
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metadata() -> dict:
    """Loads the metadata of the dataset.

    Returns: A dictionary containing the metadata.

    """
    with open(os.path.join(BASE_DIR, "data/metadata.json"), encoding="utf-8") as file:
        return json.load(file)


def load_full_dataset(binary=False) -> dict:
    """Loads the full dataset.

    Args:
        binary: Whether to load the binary dataset.

    Returns: A dictionary of dataframes containing the datasets.

    """
    metadata = load_metadata()
    dataset_type = "binary" if binary else "multiclass"
    data = {}

    for name in ["train", "test", "dev"]:
        # Load existing files if they exist
        if os.path.exists(os.path.join(BASE_DIR, f"data/{dataset_type}/{name}.json")):
            with open(os.path.join(BASE_DIR, f"data/{dataset_type}/{name}.json"), encoding="utf-8") as file:
                current_data = json.load(file)
        else:
            current_data = []
            current_dir = os.path.join(BASE_DIR, f"data/{name}/")
            for file in tqdm(os.listdir(current_dir), desc=f"Loading {dataset_type} {name} data"):
                current_file_path = os.path.join(current_dir, file)
                current_file_id = file.split(".")[0]
                with open(current_file_path, encoding="utf-8") as current_file:
                    label = metadata[current_file_id]["rating"]
                    if binary:
                        if label <= 3:
                            label = 0
                        else:
                            label = 1
                    else:
                        if label <= 2:
                            label = 0
                        elif label == 3:
                            label = 1
                        else:
                            label = 2
                    current_data.append(
                        {
                            "text": current_file.read(),
                            "label": label,
                        }
                    )
            # Save json files for later use
            with open(os.path.join(BASE_DIR, f"data/{dataset_type}/{name}.json"), "w", encoding="utf-8") as file:
                json.dump(current_data, file)

        # Convert to pandas dataframe and shuffle
        data[name] = pd.DataFrame.from_dict(current_data).sample(frac=1).reset_index(drop=True)
    return data


def load_balanced_dataset(binary=False) -> dict:
    """Loads the balanced (undersampled) dataset.

    Args:
        binary: Whether to load the binary dataset.

    Returns: A dictionary of dataframes containing the balanced (undersampled) datasets.

    """
    data = load_full_dataset(binary=binary)

    balanced_data = {}
    for df_name, df in data.items():
        if binary:
            negatives_df = df[df["label"] == 0]
            postives_df = df[df["label"] == 1]
            # Find minority class
            smallest_size = min(negatives_df.shape[0], postives_df.shape[0])
            # Undersample majority class
            negatives_df = negatives_df.sample(n=smallest_size)
            postives_df = postives_df.sample(n=smallest_size)

            result = pd.concat([negatives_df, postives_df]).reset_index(drop=True)
        else:
            negatives_df = df[df["label"] == 0]
            neutrals_df = df[df["label"] == 1]
            postives_df = df[df["label"] == 2]
            # Find minority class
            smallest_size = min(negatives_df.shape[0], neutrals_df.shape[0], postives_df.shape[0])
            # Undersample majority class
            negatives_df = negatives_df.sample(n=smallest_size)
            neutrals_df = neutrals_df.sample(n=smallest_size)
            postives_df = postives_df.sample(n=smallest_size)

            # Combine and shuffle the results
            result = pd.concat([negatives_df, neutrals_df, postives_df]).sample(frac=1).reset_index(drop=True)

        balanced_data[df_name] = result

    return balanced_data
