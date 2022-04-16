import numpy as np
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_embeddings() -> dict:
    """Loads the embeddings from the file.

    Returns: Embeddings in a dictionary.

    """

    # Load numpy embeddings from the file if it exists
    if os.path.exists(os.path.join(BASE_DIR, 'glove.4M.100d.npy')):
        return np.load(os.path.join(BASE_DIR, 'glove.4M.100d.npy'), allow_pickle=True)

    # Load embeddings from text file
    with open(os.path.join(BASE_DIR, "glove.4M.100d.txt"), encoding="utf-8") as f:
        content = f.readlines()[1:]

    # Create dictionary of embeddings
    embeddings_index = dict()
    for line in tqdm(content, desc="Loading embeddings"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

    # Save numpy embeddings to file
    np.save(os.path.join(BASE_DIR, 'glove.4M.100d.npy'), embeddings_index)

    return embeddings_index
