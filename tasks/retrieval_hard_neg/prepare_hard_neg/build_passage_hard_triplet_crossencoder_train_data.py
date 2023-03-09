import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict, deque

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


def _load_embedding_data(embedding_path):

    with open(embedding_path, "rb") as file:
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
def search_from_index_mp(args, qids_set, query_embeddings):

    print(f"start retrievaling for {len(qids_set)} queries")
    index = faiss.read_index(args.index_save_path)

    if args.nprobe > 0:
        index.nprobe = args.nprobe

    query_embeddings = {qid: query_embeddings[qid] for qid in qids_set}

    print(f"loading query embeddings of {len(query_embeddings)} queries")
    query_embeddings_np = np.array(list(query_embeddings.values()))

    def _search(i):
        dot_res, ind = index.search(query_embeddings_np[i : i + 1], args.topk)
        return ind[0].tolist()

    pool = ThreadPool(args.num_proc)
    retrieval_results = {}
    search_results = pool.map(_search, range(len(query_embeddings_np)))
    for qid_key, result in zip(query_embeddings.keys(), search_results):
        result = [str(pid) for pid in result]
        retrieval_results[qid_key] = result

    os.makedirs(os.path.dirname(args.retrival_output_path), exist_ok=True)
    with open(args.retrival_output_path, "w") as file:
        json.dump(retrieval_results, file)

    print("done retrieval!")

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

    print("building final training data!")
    qid_2_query_all = _read_collections(args.query_collection)
    pid_2_passage_all = _read_collections(args.passage_collection)

    qid_2_query = {}
    pid_2_passage = {}
    for qid, pos_pid, neg_pid in tqdm.tqdm(triplets):
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
    print(
        f"there are {len(triplets)} input triplets and distinct {len(distinct_qids)} queries!"
    )
    return triplets, distinct_qids


def gen_static_hardnegs(
    args, retrieval_results, triplets, query_embeddings, doc_embeddings
):
    rel_dict = _load_rel(args.qrel_path)

    # remove false negative
    for k in retrieval_results:
        v = retrieval_results[k]
        v = list(filter(lambda x: x not in rel_dict[k], v))
        retrieval_results[k] = v

    new_triplets = []
    existing_dot_values = []
    new_dot_values = []
    replacement_count = 0
    for triplet in tqdm.tqdm(triplets):
        qid, pid_pos, pid_neg = triplet

        existing_dot_value = np.sum(query_embeddings[qid] * doc_embeddings[pid_neg])
        existing_dot_values.append(existing_dot_value)

        ntrial = len(retrieval_results[qid])
        candidates = retrieval_results[qid]
        # If we shuffle the candidates, we can generate harder negatives
        # random.shuffle(candidates)
        candidates = deque(candidates)
        while candidates and ntrial > 0:
            hard_neg_pid = candidates.pop()
            new_dot_value = np.sum(query_embeddings[qid] * doc_embeddings[hard_neg_pid])
            if new_dot_value > existing_dot_value:
                new_triplets.append((qid, pid_pos, hard_neg_pid))
                new_dot_values.append(new_dot_value)
                replacement_count += 1
                break
            else:
                candidates.appendleft(hard_neg_pid)
            ntrial -= 1

    print(f"there are {len(new_triplets)} new triplets in the training data")
    print(
        f"there are {replacement_count} replacements: old dot value {np.mean(np.array(existing_dot_values))}, new dot value {np.mean(np.array(new_dot_values))}"
    )

    with open(os.path.join(args.output_dir, "triplets.pkl"), "wb") as file:
        pickle.dump(new_triplets, file)
    print(f"done triplets!")
    return new_triplets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--triplet_file", required=True)
    parser.add_argument("--passage_collection", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--query_embedding_path", type=str, required=True)
    parser.add_argument("--doc_embedding_path", type=str, required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--index_save_path", required=True)
    parser.add_argument("--nprobe", type=int, default=-1)
    parser.add_argument("--retrival_output_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=24)
    parser.add_argument("--name_tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    triplets, distict_qids = read_triplets(args)
    query_embeddings, doc_embeddings = _load_embedding_data(
        args.query_embedding_path
    ), _load_embedding_data(args.doc_embedding_path)
    retrieval_results = search_from_index_mp(args, distict_qids, query_embeddings)
    hard_triplets = gen_static_hardnegs(
        args, retrieval_results, triplets, query_embeddings, doc_embeddings
    )
    build_training_data(args, hard_triplets)
