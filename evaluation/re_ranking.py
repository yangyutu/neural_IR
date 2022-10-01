import torch
from utils.common import encode_plus, batch_inference
import numpy as np


def biencoder_re_rank(
    model,
    qid,
    candidate_id_list,
    tokenizer,
    qid_2_query_token_ids,
    pid_2_passage_token_ids,
    device,
    batch_size
):

    query_token = qid_2_query_token_ids[qid]
    query_encoded = encode_plus(query_token, tokenizer)
    query_encoded = query_encoded.to(device)
    query_embed = model(query_encoded).cpu().numpy()

    passage_encoded = [
        encode_plus(pid_2_passage_token_ids[pid], tokenizer)
        for pid in candidate_id_list
        if pid in pid_2_passage_token_ids
    ]

    if not passage_encoded:
        return []

    passage_embed = batch_inference(
        model, passage_encoded, device, return_in_cpu_numpy=True, batch_size=batch_size
    )

    similarities = np.sum(query_embed * np.array(passage_embed), axis=-1)

    # descending sort
    sorted_idx = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_idx]
    sorted_pids = [candidate_id_list[pid] for pid in sorted_idx]

    return zip(sorted_pids, sorted_similarities)

def crossencoder_re_rank(
    model,
    qid,
    candidate_id_list,
    tokenizer,
    qid_2_query_token_ids,
    pid_2_passage_token_ids,
    device,
    batch_size
):

    query_token = qid_2_query_token_ids[qid]
    query_encoded = encode_plus(query_token, tokenizer)
    query_encoded = query_encoded.to(device)
    query_embed = model(query_encoded).cpu().numpy()

    passage_encoded = [
        encode_plus(pid_2_passage_token_ids[pid], tokenizer)
        for pid in candidate_id_list
        if pid in pid_2_passage_token_ids
    ]

    if not passage_encoded:
        return []

    passage_embed = batch_inference(
        model, passage_encoded, device, return_in_cpu_numpy=True, batch_size=batch_size
    )

    similarities = np.sum(query_embed * np.array(passage_embed), axis=-1)

    # descending sort
    sorted_idx = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_idx]
    sorted_pids = [candidate_id_list[pid] for pid in sorted_idx]

    return zip(sorted_pids, sorted_similarities)

