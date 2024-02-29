from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import lightning as L
import pandas as pd
import numpy as np
import torch


class NoReCDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple:
        text = self.dataframe['text'].iloc[idx]
        label = self.dataframe['label'].iloc[idx]
        return text, label.astype(int)


class NoReCDataModule(L.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = NoReCDataset(self.train_df)
            self.val_dataset = NoReCDataset(self.val_df)
        elif stage == "test":
            self.test_dataset = NoReCDataset(self.test_df)

    def get_class_weights(self) -> torch.Tensor:
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['label']), y=self.train_df['label'])
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
