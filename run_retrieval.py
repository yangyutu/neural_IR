import os
import csv
from models.model_interface import PLModelTripletInterface
from models.bert_encoder import BertEncoder
from losses.triplet_loss import TripletLoss
from transformers import AutoTokenizer
import pytorch_lightning as pl
import pickle, json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from indexing.faiss_index import search_faiss_index
from argparse import ArgumentParser
from utils.common import encode_plus
import faiss


def _load_data(args):

    with open(args.qid_2_query_token_ids_path, "rb") as file:
        qid_2_query_token_ids = pickle.load(file)

    with open(args.pid_2_embedding_path, "rb") as file:
        pid_2_embedding = pickle.load(file)

    return qid_2_query_token_ids, list(pid_2_embedding.keys())


def _load_model(args):

    model = PLModelTripletInterface(**vars(args))
    model.load_from_checkpoint(args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.freeze()
    return model


def main(args):

    qid_2_query_token_ids, id_2_pid = _load_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = _load_model(args)
    index = faiss.read_index(args.index_save_path)
    retrieval_result_all = []
    count = 0
    for qid in tqdm(qid_2_query_token_ids):
        qid_encoded = encode_plus(qid_2_query_token_ids[qid], tokenizer)
        qid_embed = model(qid_encoded).numpy()
        # print(qid_embed.shape)
        retrieval_result = search_faiss_index(
            qid_embed, index, args.number_nearest_neighbors,
        )
        # print(retrieval_result)
        embed_ids, dists = retrieval_result
        for idx, result in enumerate(zip(embed_ids[0], dists[0])):
            embed_id, dist = result
            pid = id_2_pid[embed_id]
            retrieval_result_all.append((int(qid), pid, idx + 1, dist))

        # print(retrieval_result_all)

        count += 1
        # if count > 0:
        #     break

    with open(args.output_file, "w", newline="\n") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        for record in retrieval_result_all:
            writer.writerow(record)

    print("done")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--qid_2_query_token_ids_path", type=str, required=True)
    parser.add_argument("--pid_2_embedding_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--index_save_path", type=str, required=True)
    parser.add_argument("--number_nearest_neighbors", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    PLModelTripletInterface.add_model_specific_args(parser)
    BertEncoder.add_model_specific_args(parser)
    TripletLoss.add_loss_specific_args(parser)
    args = parser.parse_args()

    print(vars(args))
    main(args)
