import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def _read_collections(filename):
    with open(filename) as f:
        id_2_text = {}
        for line in f:
            id, text = line.strip().split("\t")
            id_2_text[id] = text

    logger.info(f"there are {len(id_2_text)} entries in the collection {filename}")
    return id_2_text


def build_tokenized_data(args):
    qid_2_query_all = _read_collections(args.query_collection)
    pid_2_passage_all = _read_collections(args.passage_collection)

    with open(args.query_candidates_path, "r") as file:
        query_candidates = json.load(file)

    if args.sample_size > 0:
        random.seed(1)
        queries = list(query_candidates.keys())
        random.shuffle(queries)
        queries = queries[: args.sample_size]
        query_candidates = {q: query_candidates[q] for q in queries}

    with open(
        os.path.join(args.output_dir, "qid_2_top_1000_passage_BM25_subset.json"), "w"
    ) as file:
        json.dump(query_candidates, file)

    qid_2_query = {}
    pid_2_passage = {}
    for qid, candidates in tqdm(query_candidates.items()):
        for pid in candidates:
            qid_text = qid_2_query_all[qid]
            pid_text = pid_2_passage_all[pid]
            if qid not in qid_2_query:
                qid_2_query[qid] = qid_text
            if pid not in pid_2_passage:
                pid_2_passage[pid] = pid_text

    with open(os.path.join(args.output_dir, "qid_2_query_text.pkl"), "wb") as file:
        pickle.dump(qid_2_query, file)
    with open(os.path.join(args.output_dir, "pid_2_passage_text.pkl"), "wb") as file:
        pickle.dump(pid_2_passage, file)

    logger.info(f"done!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--passage_collection", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--query_candidates_path", required=True)
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    build_tokenized_data(args)
