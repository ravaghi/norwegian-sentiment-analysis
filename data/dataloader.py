from typing import Tuple
from tqdm import tqdm
import pandas as pd
import json
import os


class NoReCDataLoader:
    def __init__(self, data_dir: str, save: bool = True, save_dir: str = None) -> None:
        self.data_dir = data_dir
        self.save = save
        self.save_dir = save_dir

    def _parse_rating(self, rating: int, binary: bool) -> int:
        if binary:
            if rating <= 3:
                return 0
            else:
                return 1
        else:
            if rating < 4:
                return 0
            elif rating == 4:
                return 1
            else:
                return 2

    def _load_metadata(self, binary: bool) -> pd.DataFrame:
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)

        data = []
        for key, value in tqdm(metadata.items(), desc="Loading metadata"):
            temp_data = value
            temp_data["text_id"] = key
            temp_data["authors"] = ", ".join(temp_data["authors"])
            temp_data["source-tags"] = ", ".join(temp_data["source-tags"])
            temp_data["tags"] = ", ".join(temp_data["tags"])
            temp_data["label"] = self._parse_rating(
                temp_data["rating"], binary)
            data.append(temp_data)

        dataset = pd.DataFrame(data)
        dataset = dataset.astype(
            {
                "authors": "category",
                "category": "category",
                "day": "int32",
                "excerpt": "string",
                "id": "int32",
                "language": "category",
                "month": "int32",
                "rating": "int32",
                "source": "category",
                "source-category": "category",
                "source-id": "string",
                "source-tags": "string",
                "split": "category",
                "tags": "string",
                "title": "string",
                "url": "string",
                "year": "int32",
                "text_id": "string",
                "label": "int32",
            }
        )
        return dataset

    def _load_documents(self) -> pd.DataFrame:
        data = []
        for name in ["train", "test", "dev"]:
            current_dir = os.path.join(self.data_dir, name)
            for file in tqdm(os.listdir(current_dir), desc=f"Loading {name} data"):
                current_file_path = os.path.join(current_dir, file)
                current_file_id = file.split(".")[0]
                with open(current_file_path, encoding="utf-8") as current_file:
                    data.append(
                        {"text_id": current_file_id, "text": current_file.read()}
                    )
        return pd.DataFrame(data)

    def _load_data(self, binary: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        file_prefix = "binary" if binary else "multiclass"
        if self.save:
            train_path = os.path.join(self.save_dir, f"{file_prefix}_train.parquet")
            val_path = os.path.join(self.save_dir, f"{file_prefix}_val.parquet")
            test_path = os.path.join(self.save_dir, f"{file_prefix}_test.parquet")

            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                print(f"Loading {file_prefix} data from disk")
                train = pd.read_parquet(train_path)
                val = pd.read_parquet(val_path)
                test = pd.read_parquet(test_path)

                return train, val, test

        print(f"Loading {file_prefix} data from source")
        metadata = self._load_metadata(binary)
        documents = self._load_documents()
        dataset = metadata.merge(documents, on="text_id")

        train = dataset[dataset["split"] == "train"]
        val = dataset[dataset["split"] == "dev"]
        test = dataset[dataset["split"] == "test"]

        if self.save:
            print("Saving data to disk")
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            train.to_parquet(train_path, engine="fastparquet")
            val.to_parquet(val_path, engine="fastparquet")
            test.to_parquet(test_path, engine="fastparquet")

        return train, val, test

    def load_binary_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._load_data(binary=True) 
    
    def load_multiclass_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._load_data(binary=False)
