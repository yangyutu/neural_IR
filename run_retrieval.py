import csv
import pickle
import time
from argparse import ArgumentParser

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from indexing.faiss_index import (
    construct_flatindex_from_embeddings,
    convert_index_to_gpu,
    index_retrieve,
)


def main(args):
    with open(args.id_2_doc_embed_filename, "rb") as file:
        id_2_doc_embeds = pickle.load(file)

    with open(args.id_2_query_embed_filename, "rb") as file:
        id_2_query_embeds = pickle.load(file)

    doc_ids, doc_embeds = zip(*id_2_doc_embeds.items())
    query_ids, query_embeds = zip(*id_2_query_embeds.items())

    start = time.time()
    index = construct_flatindex_from_embeddings(np.array(doc_embeds), np.array(doc_ids))
    if args.faiss_gpu_id >= 0:
        gpu_index = convert_index_to_gpu(
            index=index, faiss_gpu_index=args.faiss_gpu_id, useFloat16=True
        )
    print(f"index construction time: {time.time() - start}")

    nearest_nbs = index_retrieve(index, np.array(query_embeds), topk=1000, batch=128)

    with open(args.output_rank_file_path, "w") as outputfile:
        for qid, nb_list in zip(query_ids, nearest_nbs):
            for idx, nb_id in enumerate(nb_list):
                outputfile.write(f"{qid}\t{nb_id}\t{idx + 1}\n")


def parse_arguments():

    parser = ArgumentParser()
    parser.add_argument("--id_2_query_embed_filename", type=str, required=True)
    parser.add_argument("--id_2_doc_embed_filename", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--faiss_gpu_id", type=int, default=-1)
    parser.add_argument(
        "--output_rank_file_path", type=str, default="./retrieval_rank.tsv"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
