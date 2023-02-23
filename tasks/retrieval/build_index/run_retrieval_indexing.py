import pickle
from argparse import ArgumentParser
import numpy as np
import faiss
import os


def _load_embedding_data(args):

    with open(args.doc_embedding_path, "rb") as file:
        doc_embeddings = pickle.load(file)

    print(f"load passages {len(doc_embeddings)} !")

    return doc_embeddings


def build_exact_search_index(args):

    doc_embeddings = _load_embedding_data(args)
    dim = len(next(iter(doc_embeddings.values())))
    print(f"embedding dim {dim}")
    # following the guide to build an index with doc vector ids
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    doc_embeddings_np = np.array(list(doc_embeddings.values()))
    doc_embeddings_id_np = np.array(list(map(int, list(doc_embeddings.keys())))).astype(
        np.int64
    )
    print(doc_embeddings_np.shape)
    print(doc_embeddings_id_np.shape)
    index.add_with_ids(doc_embeddings_np, doc_embeddings_id_np)

    print(f"number of vectors indexed: {index.ntotal}")

    os.makedirs(os.path.dirname(args.index_save_path), exist_ok=True)
    faiss.write_index(index, args.index_save_path)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--doc_embedding_path", type=str, required=True)
    # parser.add_argument("--exact_search", action="store_true")
    parser.add_argument("--index_save_path", type=str, required=True)

    args = parser.parse_args()

    build_exact_search_index(args)
