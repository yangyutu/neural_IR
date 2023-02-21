import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer


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
    with open(args.triplet_file) as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split("\t")
            triplets.append((qid, pos_id, neg_id))
            count += 1

    print(f"there are {count} triplets in the training data")
    with open(os.path.join(args.output_dir, "triplets.pkl"), "wb") as file:
        pickle.dump(triplets, file)
    print(f"done triplets!")

    return triplets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--triplet_file", required=True)
    parser.add_argument("--passage_collection", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--name_tag", type=str, default="")
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    triplets = read_triplets(args)
    build_training_data(args, triplets)
