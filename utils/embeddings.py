import numpy as np
import os
from tqdm import tqdm
import requests
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_embedding() -> np.ndarray:
    """Downloads the embeddings from the url.

    Returns: A list of the embeddings.

    """
    url = "http://vectors.nlpl.eu/repository/20/79.zip"
    file_name = "79.zip"
    file_path = os.path.join(BASE_DIR, file_name)

    print("Downloading embeddings ...")
    with open(file_path, "wb") as f:
        f.write(requests.get(url).content)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        print("Extracting embeddings...")
        zip_ref.extractall(BASE_DIR)

    # Remove zip file and other files
    os.remove(file_path)
    os.remove(os.path.join(BASE_DIR, "README"))
    os.remove(os.path.join(BASE_DIR, "model.bin"))
    os.remove(os.path.join(BASE_DIR, "meta.json"))
    os.rename(os.path.join(BASE_DIR, "model.txt"), os.path.join(BASE_DIR, "glove.4M.100d.txt"))

    # Read text file
    with open(os.path.join(BASE_DIR, "glove.4M.100d.txt"), encoding="utf-8") as file:
        content = file.readlines()[1:]

    # Create dictionary of embeddings
    embeddings_index = dict()
    for line in tqdm(content, desc="Loading embeddings"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

    # Save numpy embeddings to file
    np.save(os.path.join(BASE_DIR, 'glove.4M.100d.npy'), embeddings_index)

    os.remove(os.path.join(BASE_DIR, "glove.4M.100d.txt"))

    return np.load(os.path.join(BASE_DIR, 'glove.4M.100d.npy'), allow_pickle=True)


def load_embeddings() -> np.ndarray:
    """Loads the embeddings from the file.

    Returns: Embeddings in a dictionary.

    """

    # Load numpy embeddings from the file if it exists
    if os.path.exists(os.path.join(BASE_DIR, 'glove.4M.100d.npy')):
        return np.load(os.path.join(BASE_DIR, 'glove.4M.100d.npy'), allow_pickle=True)

    return download_embedding()
