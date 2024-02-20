from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from typing import Tuple, Callable, List
from torchtext.vocab import Vocab
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import re
import os


class NoReCDataPreprocessor:
    @staticmethod
    def convert_to_lowercase(dataframe: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Converting to lowercase")
        dataframe.loc[:, "text"] = dataframe["text"].progress_apply(
            lambda x: x.lower()
        )
        return dataframe

    @staticmethod
    def remove_punctuation(dataframe: pd.DataFrame) -> pd.DataFrame:
        def clean(text):
            text = text.replace("\n", " ")
            text = re.sub(r'[^a-zæøå]+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text

        tqdm.pandas(desc="Removing punctuation")
        dataframe.loc[:, "text"] = dataframe["text"].progress_apply(lambda x: clean(x))
        
        return dataframe

    @staticmethod
    def load_stopwords() -> List[str]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stopwords.json")
        with open(path, encoding="utf-8") as file:
            stopwords = json.load(file)
            
        return stopwords

    def remove_stopwords(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        stopwords = self.load_stopwords()

        tqdm.pandas(desc="Removing stopwords")
        dataframe.loc[:, "text"] = dataframe["text"].progress_apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in stopwords]
            )
        )
        return dataframe

    def sanitize(self, dataframe: pd.DataFrame, type: str) -> pd.DataFrame:
        print(f"\nProcessing {type} dataframe...")
        dataframe = self.convert_to_lowercase(dataframe)
        dataframe = self.remove_punctuation(dataframe)
        dataframe = self.remove_stopwords(dataframe)
        return dataframe[["text", "label"]]

    def build_vocabulary(self, dataframe: pd.DataFrame, vocab_size: int) -> Tuple[Vocab, Callable]:
        df_iterator = list(dataframe["text"])
        tokenizer = get_tokenizer(tokenizer="spacy", language="nb_core_news_lg")

        def tokenizer_fn(data_iterator):
            for text in tqdm(data_iterator, desc="Building vocabulary"):
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(
            tokenizer_fn(df_iterator),
            specials=["<unk>"],
            max_tokens=vocab_size
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab, tokenizer

    def tokenize(self, dataframe: pd.DataFrame, vocab: Vocab, tokenizer: Callable) -> pd.DataFrame:
        tqdm.pandas(desc="Tokenizing")
        dataframe.loc[:, "text"] = dataframe["text"].progress_apply(
            lambda x: np.array(vocab(tokenizer(x)), dtype=np.int64)
        )
        return dataframe

    def pad(self, dataframe: pd.DataFrame, vocab: Vocab, max_seq_len: int) -> pd.DataFrame:
        tqdm.pandas(desc="Padding")
        dataframe.loc[:, 'text'] = dataframe['text'].progress_apply(
            lambda x: np.pad(x, (0, max(0, max_seq_len - len(x))),
                             'constant', constant_values=vocab["<unk>"])
        )
        dataframe.loc[:, "text"] = dataframe["text"].apply(
            lambda x: x[:max_seq_len]
        )
        return dataframe
