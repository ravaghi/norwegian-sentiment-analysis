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


def load_full_dataset() -> dict:
    """Loads the full dataset.

    Returns: A dictionary of dataframes containing the multiclass datasets.

    """
    metadata = load_metadata()
    data = {}
    for name in ["train", "test", "dev"]:
        if os.path.exists(os.path.join(BASE_DIR, f"data/multiclass/{name}.json")):
            with open(os.path.join(BASE_DIR, f"data/multiclass/{name}.json"), encoding="utf-8") as file:
                current_data = json.load(file)
        else:
            current_data = []
            current_dir = os.path.join(BASE_DIR, f"data/{name}/")
            for file in tqdm(os.listdir(current_dir), desc=f"Loading multiclass {name} data"):
                current_file_path = os.path.join(current_dir, file)
                current_file_id = file.split(".")[0]
                with open(current_file_path, encoding="utf-8") as current_file:
                    label = metadata[current_file_id]["rating"]
                    if label <= 2:
                        label = 0
                    elif 2 < label <= 4:
                        label = 1
                    else:
                        label = 2
                    current_data.append(
                        {
                            "text": current_file.read(),
                            "label": label,
                        }
                    )
            with open(os.path.join(BASE_DIR, f"data/multiclass/{name}.json"), "w", encoding="utf-8") as file:
                json.dump(current_data, file)
        # Convert to pandas dataframe
        df = pd.DataFrame(current_data)
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
    metadata = load_metadata()
    data = {}
    for name in ["train", "test", "dev"]:
        if os.path.exists(os.path.join(BASE_DIR, f"data/binary/{name}.json")):
            with open(os.path.join(BASE_DIR, f"data/binary/{name}.json"), encoding="utf-8") as file:
                current_data = json.load(file)
        else:
            current_data = []
            current_dir = os.path.join(BASE_DIR, f"data/{name}/")
            for file in tqdm(os.listdir(current_dir), desc=f"Loading binary {name} data"):
                current_file_path = os.path.join(current_dir, file)
                current_file_id = file.split(".")[0]
                with open(current_file_path, encoding="utf-8") as current_file:
                    label = metadata[current_file_id]["rating"]
                    if label <= 3:
                        label = 0
                    else:
                        label = 1
                    current_data.append(
                        {
                            "text": current_file.read(),
                            "label": label,
                        }
                    )
            with open(os.path.join(BASE_DIR, f"data/binary/{name}.json"), "w", encoding="utf-8") as file:
                json.dump(current_data, file)
        # Convert to pandas dataframe
        df = pd.DataFrame(current_data)
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

    return balanced_data
