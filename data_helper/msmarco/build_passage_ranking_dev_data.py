from argparse import ArgumentParser
from transformers import AutoTokenizer

from tqdm import tqdm
import json, os, pickle
from collections import defaultdict
import logging
import random

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
    qid_2_query = _read_collections(args.query_collection)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    qid_2_query_token_ids = {}

    for qid in tqdm(qid_2_query):
        text = qid_2_query[qid]
        qid_2_query_token_ids[qid] = tokenizer.encode(
            text, add_special_tokens=False, max_length=args.truncate, truncation=True
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "qid_2_query_token_ids.pkl"), "wb") as file:
        pickle.dump(qid_2_query_token_ids, file)

    logger.info(f"done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--query_collection", required=True)
    parser.add_argument("--truncate", type=int, default=128)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    print(args)

    build_tokenized_data(args)

