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


def _load_rel(rel_path):
    reldict = defaultdict(list)
    with open(rel_path, "r") as file:
        for line in file:
            qid, _, pid, _ = line.split()
            qid, pid = qid, pid
            reldict[qid].append((pid))
    return dict(reldict)


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
    distinct_pids = set()
    with open(args.triplet_file) as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split("\t")
            distinct_qids.add(qid)
            distinct_pids.add(pos_id)
            distinct_pids.add(neg_id)
            triplets.append((qid, pos_id, neg_id))
            count += 1
    print(
        f"there are {len(triplets)} input triplets and distinct {len(distinct_qids)} queries!"
    )
    return triplets, distinct_qids, distinct_pids


def gen_simple_negs(args, distinct_qids, distinct_pids, triplets):
    rel_dict = _load_rel(args.qrel_path)

    retrieval_results = {}
    np.random.seed(args.seed)
    distinct_pids = list(distinct_pids)
    for qid in tqdm.tqdm(distict_qids):
        rand_array = np.random.randint(0, len(distinct_pids), args.topk)
        retrieval_results[qid] = [distinct_pids[i] for i in rand_array]

    # remove false negative
    for k in retrieval_results:
        v = retrieval_results[k]
        v = list(filter(lambda x: x not in rel_dict[k], v))
        retrieval_results[k] = v

    new_triplets = []
    for triplet in tqdm.tqdm(triplets):
        qid, pid_pos, pid_neg = triplet

        if retrieval_results[qid]:
            simple_neg = retrieval_results[qid].pop()
            new_triplets.append((qid, pid_pos, simple_neg))

    print(f"there are {len(new_triplets)} new triplets in the training data")
    with open(os.path.join(args.output_dir, "triplets.pkl"), "wb") as file:
        pickle.dump(new_triplets, file)
    print(f"done triplets!")
    return new_triplets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--triplet_file", required=True)
    parser.add_argument("--passage_collection", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--name_tag", type=str, default="")

    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    triplets, distict_qids, distinct_pids = read_triplets(args)
    simple_triplets = gen_simple_negs(args, distict_qids, distinct_pids, triplets)
    build_training_data(args, simple_triplets)
