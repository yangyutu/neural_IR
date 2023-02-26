import pickle
from argparse import ArgumentParser
import numpy as np
import faiss
import os


def _load_embedding_data(args):

    with open(args.doc_embedding_path, "rb") as file:
        doc_embeddings = pickle.load(file)

    print(f"load documents {len(doc_embeddings)} !")

    return doc_embeddings


def build_exact_search_index(args):

    doc_embeddings = _load_embedding_data(args)
    dim = len(next(iter(doc_embeddings.values())))
    print(f"embedding dim {dim}")
    # following the guide to build an index with doc vector ids
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
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


def build_ivf_index(args):

    doc_embeddings = _load_embedding_data(args)
    dim = len(next(iter(doc_embeddings.values())))
    print(f"embedding dim {dim}")
    # following the guide to build an index with doc vector ids
    quantizer = faiss.IndexFlatIP(dim)  # vector quantizer
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT)
    doc_embeddings_np = np.array(list(doc_embeddings.values()))
    doc_embeddings_id_np = np.array(list(map(int, list(doc_embeddings.keys())))).astype(
        np.int64
    )
    print(doc_embeddings_np.shape)
    print(doc_embeddings_id_np.shape)
    np.random.seed(args.seed)
    sampled_row_ids = np.random.choice(
        doc_embeddings_np.shape[0],
        int(args.data_frac_for_training * doc_embeddings_np.shape[0]),
        replace=False,
    )
    training_data = doc_embeddings_np[sampled_row_ids]
    index.train(training_data)
    index.add_with_ids(doc_embeddings_np, doc_embeddings_id_np)

    print(f"number of vectors indexed: {index.ntotal}")

    os.makedirs(os.path.dirname(args.index_save_path), exist_ok=True)
    faiss.write_index(index, args.index_save_path)


def build_ivfpq_index(args):

    doc_embeddings = _load_embedding_data(args)
    dim = len(next(iter(doc_embeddings.values())))
    print(f"embedding dim {dim}")
    # following the guide to build an index with doc vector ids

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        args.nlist,
        args.subquantizer_number,
        args.subquantizer_codebook_size,
    )
    doc_embeddings_np = np.array(list(doc_embeddings.values()))
    doc_embeddings_id_np = np.array(list(map(int, list(doc_embeddings.keys())))).astype(
        np.int64
    )
    np.random.seed(args.seed)
    sampled_row_ids = np.random.choice(
        doc_embeddings_np.shape[0],
        int(args.data_frac_for_training * doc_embeddings_np.shape[0]),
        replace=False,
    )
    training_data = doc_embeddings_np[sampled_row_ids]
    index.train(training_data)
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
    parser.add_argument("--index_type", type=str, choices=["exact", "ivf", "ivfpq"])
    parser.add_argument("--data_frac_for_training", type=float, default=0.1)
    parser.add_argument("--nlist", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--subquantizer_number", type=int, default=8)
    parser.add_argument("--subquantizer_codebook_size", type=int, default=8)

    args = parser.parse_args()

    if args.index_type == "exact":
        build_exact_search_index(args)
    elif args.index_type == "ivf":
        build_ivf_index(args)
    elif args.index_type == "ivfpq":
        build_ivfpq_index(args)
    else:
        raise ValueError("index type not allowed")
