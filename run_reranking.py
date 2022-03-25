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
from evaluation.re_ranking import re_rank
from argparse import ArgumentParser


def _load_data(args):

    with open(args.qid_2_query_token_ids_path, "rb") as file:
        qid_2_query_token_ids = pickle.load(file)

    with open(args.pid_2_passage_token_ids_path, "rb") as file:
        pid_2_passage_token_ids = pickle.load(file)

    with open(args.re_rank_input_file_path, "r") as file:
        query_2_candidates = json.load(file)

    return qid_2_query_token_ids, pid_2_passage_token_ids, query_2_candidates


def _load_model(args, device):

    model = PLModelTripletInterface(**vars(args))
    model.load_from_checkpoint(args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.freeze()
    model.to(device)
    return model


def main(args):

    qid_2_query_token_ids, pid_2_passage_token_ids, query_2_candidates = _load_data(
        args
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = _load_model(args, device)
    re_rank_result_all = []
    count = 0
    for qid, candidate_list in tqdm(query_2_candidates.items()):
        if qid not in qid_2_query_token_ids:
            continue
        re_rank_result = re_rank(
            model,
            qid,
            candidate_list,
            tokenizer,
            qid_2_query_token_ids,
            pid_2_passage_token_ids,
            device,
        )

        for idx, result in enumerate(re_rank_result):
            pid, score = result
            re_rank_result_all.append((int(qid), pid, idx + 1, score))

        count += 1

    with open(args.output_file, "w", newline="\n") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        for record in re_rank_result_all:
            writer.writerow(record)


def verify():
    model1 = PLModelTripletInterface(
        model_name="bert_encoder",
        loss_name="triplet_loss",
        lr=1e-6,
        lr_scheduler=None,
        pretrained_model_name="distilbert-base-uncased",
        margin=20.0,
    )
    # model1.load_from_checkpoint(
    #     "/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M/checkpoints/epoch=2-step=93749.ckpt"
    # )
    path1 = "/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M_margin20/checkpoints/epoch=1-step=31249.ckpt"
    model1.load_from_checkpoint(
        "/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M_margin20/checkpoints/epoch=1-step=31249.ckpt"
    )
    checkpoint1 = torch.load(path1)
    model1.load_state_dict(checkpoint1["state_dict"])

    model2 = PLModelTripletInterface(
        model_name="bert_encoder",
        loss_name="triplet_loss",
        lr=1e-6,
        lr_scheduler=None,
        pretrained_model_name="distilbert-base-uncased",
        margin=20.0,
    )
    path2 = "/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M_margin20/checkpoints/epoch=2-step=46874.ckpt"
    model2.load_from_checkpoint(
        "/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M_margin20/checkpoints/epoch=2-step=46874.ckpt"
    )
    checkpoint2 = torch.load(path2)
    model2.load_state_dict(checkpoint2["state_dict"])
    tokenizer_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    text = "Hello, world!"

    out1 = model1(
        tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation="longest_first",
            padding="max_length",
        )
    )
    out2 = model2(
        tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation="longest_first",
            padding="max_length",
        )
    )

    count = 0
    for name, param in model1.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            count += 1
            if count > 2:
                break

    count = 0
    for name, param in model2.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            count += 1
            if count > 2:
                break

    checkpoint1 = torch.load(path1)
    checkpoint2 = torch.load(path2)

    print("done")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--qid_2_query_token_ids_path", type=str, required=True)
    parser.add_argument("--pid_2_passage_token_ids_path", type=str, required=True)
    parser.add_argument("--re_rank_input_file_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    PLModelTripletInterface.add_model_specific_args(parser)
    BertEncoder.add_model_specific_args(parser)
    TripletLoss.add_loss_specific_args(parser)
    args = parser.parse_args()

    print(vars(args))
    main(args)
