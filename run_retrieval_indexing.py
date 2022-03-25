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
from indexing.faiss_index import build_faiss_index
from utils.common import encode_plus, batch_inference
from argparse import ArgumentParser
import numpy as np


def faiss_indexing(args):

    with open(args.embedding_save_path, "rb") as file:
        data = pickle.load(file)

    embeddings = np.array(list(data.values())).astype("float32")

    build_faiss_index(
        embeddings,
        embeddings.shape[1],
        args.num_partitions,
        args.subquantizer_number,
        args.subquantizer_codebook_size,
        args.index_save_path,
    )


def _load_data(args):

    with open(args.pid_2_passage_token_ids_path, "rb") as file:
        pid_2_passage_token_ids = pickle.load(file)

    print(f"load passages {len(pid_2_passage_token_ids)} !")

    return pid_2_passage_token_ids


def _load_model(args, device):

    model = PLModelTripletInterface(**vars(args))
    model.load_from_checkpoint(args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.freeze()
    model.to(device)
    return model


def compute_passage_encodings(args):

    pid_2_passage_token_ids = _load_data(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = _load_model(args, device)

    pids = list(pid_2_passage_token_ids.keys())
    passage_encoded = [
        encode_plus(pid_2_passage_token_ids[pid], tokenizer) for pid in tqdm(pids)
    ]

    if not passage_encoded:
        return

    passage_embed = batch_inference(
        model, passage_encoded, device, return_in_cpu_numpy=True
    )

    pid_2_passage_embedding = dict(zip(pids, passage_embed))

    with open(args.embedding_save_path, "wb") as file:
        pickle.dump(pid_2_passage_embedding, file)

    print("done")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--pid_2_passage_token_ids_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--embedding_save_path", type=str, required=True)

    parser.add_argument("--num_partitions", type=int, required=True)
    parser.add_argument("--subquantizer_number", type=int, required=True)
    parser.add_argument("--subquantizer_codebook_size", type=int, required=True)
    parser.add_argument("--index_save_path", type=str, required=True)

    PLModelTripletInterface.add_model_specific_args(parser)
    BertEncoder.add_model_specific_args(parser)
    TripletLoss.add_loss_specific_args(parser)
    args = parser.parse_args()

    print(vars(args))

    compute_passage_encodings(args)
    faiss_indexing(args)

