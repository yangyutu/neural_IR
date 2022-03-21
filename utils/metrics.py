from typing import Dict, List




def compute_recall_at_k(query_2_top_pid_groundtruth: Dict[int, List[int]], query_2_top_pid_candiates: Dict[int, List[int]], retrieval_num):
    recall_at_k = [len(set.intersection(set(query_2_top_pid_groundtruth[qid]), set(query_2_top_pid_candiates[qid][:retrieval_num]))) / max(1.0, len(query_2_top_pid_groundtruth[qid]))
                    for qid in query_2_top_pid_groundtruth]
    recall_at_k = sum(recall_at_k) / len(query_2_top_pid_groundtruth)
    recall_at_k = round(recall_at_k, 5)

    return recall_at_k

def compute_MRR_metrics(qids_to_relevant_passageids: Dict[int, List[int]], qids_to_ranked_candidate_passages: Dict[int, List[int]], MaxMRRRank:int = 10, verbose: bool=False):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        MRR score float
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            if len(candidate_pid) < MaxMRRRank and verbose:
                print(f"warning: candidate pid length {len(candidate_pid)} of qid {qid} is smaller than max MRR rank {MaxMRRRank}")
            for i in range(0, min(MaxMRRRank, len(candidate_pid))):
                if candidate_pid[i] in target_pid:
                    MRR += 1/(i + 1)
                    ranking.pop()
                    ranking.append(i+1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    # average over all queries
    MRR = MRR/len(qids_to_relevant_passageids)
    all_scores[f'MRR @ {MaxMRRRank}'] = MRR
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    print(all_scores)
    return MRR