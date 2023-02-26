import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

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


def _load_rel(rel_path):
    reldict = defaultdict(list)
    with open(rel_path, "r") as file:
        for line in file:
            qid, _, pid, _ = line.split()
            qid, pid = qid, pid
            reldict[qid].append((pid))
    return dict(reldict)


# multi-thread parallel via multiprocessing.dummy
# https://stackoverflow.com/questions/26432411/multiprocessing-dummy-in-python-is-not-utilising-100-cpu
# https://github.com/facebookresearch/faiss/issues/924
def search_from_index_mp(args, qids_set):

    print(f"start retrievaling for {len(qids_set)} queries")
    index = faiss.read_index(args.index_save_path)
    query_embeddings = _load_embedding_data(args)

    query_embeddings = {qid: query_embeddings[qid] for qid in qids_set}

    print(f"loading query embeddings of {len(query_embeddings)} queries")
    query_embeddings_np = np.array(list(query_embeddings.values()))

    def _search(i):
        dis, ind = index.search(query_embeddings_np[i : i + 1], args.topk)
        return ind[0].tolist()

    pool = ThreadPool(args.num_proc)
    retrieval_results = {}
    search_results = pool.map(_search, range(len(query_embeddings_np)))
    for qid_key, result in zip(query_embeddings.keys(), search_results):
        retrieval_results[qid_key] = result

    os.makedirs(os.path.dirname(args.retrival_output_path), exist_ok=True)
    with open(args.retrival_output_path, "w") as file:
        json.dump(retrieval_results, file)

    return retrieval_results


def _read_collections(filename):
    with open(filename) as f:
        id_2_text = {}
        for line in f:
            id, text = line.strip().split("\t")
            id_2_text[id] = text

    print(f"there are {len(id_2_text)} entries in the collection {filename}")
    return id_2_text


def build_training_data(args, triplets):
    qid_2_query_all = _read_collections(args.query_collection)
    pid_2_passage_all = _read_collections(args.passage_collection)

    qid_2_query = {}
    pid_2_passage = {}
    for qid, pos_pid, neg_pid in tqdm(triplets):
        qid_text = qid_2_query_all[qid]
        pos_pid_text = pid_2_passage_all[pos_pid]
        neg_pid_text = pid_2_passage_all[neg_pid]
        if qid not in qid_2_query:
            qid_2_query[qid] = qid_text
        if pos_pid not in pid_2_passage:
            pid_2_passage[pos_pid] = pos_pid_text
        if neg_pid not in pid_2_passage:
            pid_2_passage[neg_pid] = neg_pid_text

    with open(os.path.join(args.output_dir, "qid_2_query_text.pkl"), "wb") as file:
        pickle.dump(qid_2_query, file)
    with open(os.path.join(args.output_dir, "pid_2_passage_text.pkl"), "wb") as file:
        pickle.dump(pid_2_passage, file)

    print(f"done!")


def read_triplets(args):

    triplets = []
    count = 0
    distinct_qids = set()
    with open(args.triplet_file) as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split("\t")
            distinct_qids.add(qid)
            triplets.append((qid, pos_id, neg_id))
            count += 1

    return triplets, distinct_qids


def gen_static_hardnegs(args, retrieval_results, triplets):
    rel_dict = _load_rel(args.qrel_path)

    # remove false negative
    for k in retrieval_results:
        v = retrieval_results[k]
        v = list(filter(lambda x: x not in rel_dict[k], v))
        retrieval_results[k] = v

    new_triplets = []
    for triplet in tqdm.tqdm(triplets):
        qid, pid_pos, pid_neg = triplet

        if retrieval_results[qid]:
            hard_neg = retrieval_results[qid].pop()
            new_triplets.append((qid, pid_pos, hard_neg))

    print(f"there are {len(new_triplets)} new triplets in the training data")
    with open(os.path.join(args.output_dir, "hard_neg_triplets.pkl"), "wb") as file:
        pickle.dump(new_triplets, file)
    print(f"done triplets!")
    return new_triplets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--triplet_file", required=True)
    parser.add_argument("--passage_collection", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--query_embedding_path", type=str, required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--index_save_path", required=True)
    parser.add_argument("--retrival_output_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=24)
    parser.add_argument("--name_tag", type=str, default="")
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    triplets, distict_qids = read_triplets(args)
    retrieval_results = search_from_index_mp(args, distict_qids)
    hard_triplets = gen_static_hardnegs(args, retrieval_results, triplets)
    build_training_data(args, triplets)
