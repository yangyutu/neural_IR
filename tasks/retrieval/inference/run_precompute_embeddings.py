from copyreg import pickle
import os
import math
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from collections import namedtuple, defaultdict
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
import pickle
from models.bi_encoder_finetune import BiEncoderFineTune
from argparse import ArgumentParser


def _read_collections(filename):
    with open(filename) as f:
        id_2_text = {}
        for line in f:
            id, text = line.strip().split("\t")
            id_2_text[id] = text

    print(f"there are {len(id_2_text)} entries in the collection {filename}")
    return id_2_text


class MSMARCO_TextDataset(Dataset):
    def __init__(self, filename):
        self.id_2_text = _read_collections(filename)
        self.ids = list(self.id_2_text.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ret_val = {"id": self.ids[idx], "text": self.id_2_text[self.ids[idx]]}
        return ret_val


def _generate_embeddings(
    input_filename, output_filename, model, batch_size, num_workers=24, device="cuda"
):

    text_dataset = MSMARCO_TextDataset(input_filename)

    dataloader = DataLoader(
        text_dataset, batch_size=batch_size, num_workers=num_workers
    )

    print("Num examples = %d", len(text_dataset))
    print("Batch size = %d", batch_size)

    start = timer()
    id_2_embeds = {}
    model.to(device)
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            if args.query:
                output = model.compute_query_embeddings(batch["text"])
            else:
                output = model.compute_doc_embeddings(batch["text"])
            sequence_embeddings = output.detach().cpu().numpy()
        for id, embed in zip(batch["id"], sequence_embeddings):
            id_2_embeds[id] = embed
    end = timer()
    print("time:", end - start)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "wb") as file:
        pickle.dump(id_2_embeds, file)


def compute_embeddings(args):
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiEncoderFineTune.load_from_checkpoint(args.model_checkpoint)

    _generate_embeddings(
        args.input_filename,
        args.output_filename,
        model,
        args.batch_size,
        args.num_workers,
        device,
    )


def parse_arguments():

    parser = ArgumentParser()
    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--truncate", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    compute_embeddings(args)
