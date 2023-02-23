import sys
import statistics
from typing import Dict, List
from collections import Counter
import json


def compute_mrr_metrics(
    qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MaxMRRRank: int = 10
):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            # convert all elements to strings
            target_pid = list(map(str, qids_to_relevant_passageids[str(qid)]))
            candidate_pid = list(map(str, qids_to_ranked_candidate_passages[str(qid)]))
            for i in range(0, min(len(candidate_pid), MaxMRRRank)):
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError(
            "No matching QIDs found. Are you sure you are scoring the evaluation set?"
        )

    MRR = MRR / len(qids_to_relevant_passageids)
    all_scores["MRR @10"] = MRR
    all_scores["QueriesRanked"] = len(qids_to_ranked_candidate_passages)
    return all_scores


def compute_recall_at_k(
    query_2_top_pid_groundtruth: Dict[int, List[int]],
    query_2_top_pid_candiates: Dict[int, List[int]],
    retrieval_num,
):
    recall_at_k = [
        len(
            set.intersection(
                set(map(str, query_2_top_pid_groundtruth[qid])),
                set(map(str, query_2_top_pid_candiates[qid][:retrieval_num])),
            )
        )
        / max(1.0, len(query_2_top_pid_groundtruth[str(qid)]))
        for qid in query_2_top_pid_groundtruth
    ]
    recall_at_k = sum(recall_at_k) / len(query_2_top_pid_groundtruth)
    recall_at_k = round(recall_at_k, 5)

    return recall_at_k


if __name__ == "__main__":
    ground_truth_path = "/mnt/d/MLData/Repos/neural_IR/assets/msmarco/query_2_groundtruth_passage_small.json"
    candidate_path = (
        "/mnt/d/MLData/Repos/neural_IR/experiments/retrieval_results/out.json"
    )
    # candidate_path = "/mnt/d/MLData/Repos/neural_IR/assets/msmarco/query_2_top_1000_passage_BM25.json"
    with open(ground_truth_path, "r") as file:
        qids_to_relevant_passageids = json.load(file)

    with open(candidate_path, "r") as file:
        qids_to_ranked_candidate_passages = json.load(file)

    result = compute_mrr_metrics(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages
    )
    print(result)

    recall_20 = compute_recall_at_k(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 20
    )
    print(recall_20)
