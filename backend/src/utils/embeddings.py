import numpy as np
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_embeddings() -> dict:
    """Loads the embeddings from the file.

    Returns: Embeddings in a dictionary.

    """
    with open(os.path.join(BASE_DIR, "glove.4M.100d.txt"), encoding="utf-8") as f:
        content = f.readlines()[1:]

    embeddings_index = dict()
    for line in tqdm(content, desc="Loading embeddings"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

    return embeddings_index
