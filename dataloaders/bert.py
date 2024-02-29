from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List
import lightning as L
import pandas as pd
import numpy as np
import torch


class NoReCDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_seq_len: int, n_classes: int):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict:
        text = self.dataframe['text'].iloc[idx]
        labels = self.dataframe['label'].iloc[idx]

        labels = [0] * self.n_classes
        labels[self.dataframe['label'].iloc[idx]] = 1

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class NoReCDataModule(L.LightningDataModule):
    def __init__(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        batch_size: int,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        n_classes: int
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = NoReCDataset(self.train_df, self.tokenizer, self.max_seq_len, self.n_classes)
            self.val_dataset = NoReCDataset(self.val_df, self.tokenizer, self.max_seq_len, self.n_classes)
        elif stage == "test":
            self.test_dataset = NoReCDataset(self.test_df, self.tokenizer, self.max_seq_len, self.n_classes)

    def get_class_weights(self) -> torch.Tensor:
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['label']), y=self.train_df['label'])
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)