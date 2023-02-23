import pickle
from argparse import ArgumentParser
import numpy as np
import faiss
import os
import tqdm
from tqdm import trange
import json
from multiprocessing.dummy import Pool as ThreadPool


def _load_embedding_data(args):

    with open(args.query_embedding_path, "rb") as file:
        embeddings = pickle.load(file)

    print(f"load query {len(embeddings)} !")

    return embeddings


# multi-thread parallel via multiprocessing.dummy
# https://stackoverflow.com/questions/26432411/multiprocessing-dummy-in-python-is-not-utilising-100-cpu
# https://github.com/facebookresearch/faiss/issues/924
def search_from_index_mp(args):

    index = faiss.read_index(args.index_save_path)
    query_embeddings = _load_embedding_data(args)
    query_embeddings_np = np.array(list(query_embeddings.values()))

    def _search(i):
        dis, ind = index.search(query_embeddings_np[i : i + 1], args.topk)
        return ind[0].tolist()

    pool = ThreadPool(args.num_proc)
    results = {}
    search_results = pool.map(_search, range(len(query_embeddings_np)))
    for key, result in zip(query_embeddings.keys(), search_results):
        results[key] = result

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(results, file)


def search_from_index(args):

    index = faiss.read_index(args.index_save_path)
    query_embeddings = _load_embedding_data(args)

    results = {}
    count = 0
    for qid, query_embedding in tqdm.tqdm(query_embeddings.items()):
        dis, ind = index.search(query_embedding[np.newaxis, :], args.topk)
        results[qid] = ind[0].tolist()
        count += 1
        if count > 100:
            break

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--query_embedding_path", type=str, required=True)
    # parser.add_argument("--exact_search", action="store_true")
    parser.add_argument("--index_save_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--num_proc", type=int, default=24)

    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    search_from_index_mp(args)
