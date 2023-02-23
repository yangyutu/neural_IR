import pickle
from argparse import ArgumentParser
import numpy as np
import faiss
import os
import tqdm
from tqdm import trange
import json
from multiprocessing.dummy import Pool as ThreadPool
from eval_utils.ranking_metrics import compute_mrr_metrics, compute_recall_at_k


def eval(args):

    # candidate_path = "/mnt/d/MLData/Repos/neural_IR/assets/msmarco/query_2_top_1000_passage_BM25.json"
    with open(args.ground_truth_path, "r") as file:
        qids_to_relevant_passageids = json.load(file)

    with open(args.candidate_path, "r") as file:
        qids_to_ranked_candidate_passages = json.load(file)

    result = compute_mrr_metrics(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages
    )
    print(f"mrr: {result}")

    recall_5 = compute_recall_at_k(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 5
    )
    print(f"recall_5: {recall_5}")

    recall_20 = compute_recall_at_k(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 20
    )
    print(f"recall_20: {recall_20}")

    recall_100 = compute_recall_at_k(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 100
    )
    print(f"recall_100: {recall_100}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--ground_truth_path", type=str, required=True)
    # parser.add_argument("--exact_search", action="store_true")
    parser.add_argument("--candidate_path", type=str, required=True)

    args = parser.parse_args()

    eval(args)
