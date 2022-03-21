from argparse import ArgumentParser
from transformers import AutoTokenizer

from tqdm import tqdm
import json, os, pickle
from collections import defaultdict
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def _read_collections(filename):
    with open(filename) as f:
        id_2_text = {}
        for line in f:
            id, text = line.strip().split('\t')
            id_2_text[id] = text
    
    logger.info(f"there are {len(id_2_text)} entries in the collection {filename}")
    return id_2_text


def build_tokenized_data(args, qid_set, pid_set):
    qid_2_query = _read_collections(args.query_collection)
    pid_2_passage = _read_collections(args.passage_collection)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    qid_2_query_token_ids = {}
    pid_2_passage_token_ids = {} 

    for qid in tqdm(qid_2_query):
        text = qid_2_query[qid]
        qid_2_query_token_ids[qid] = tokenizer.encode(text, add_special_tokens=False, max_length=args.truncate, truncation=True)
        
    for pid in tqdm(pid_2_passage):
        if pid not in pid_set:
            continue
        text = pid_2_passage[pid]
        pid_2_passage_token_ids[pid] = tokenizer.encode(text, add_special_tokens=False, max_length=args.truncate, truncation=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'qid_2_query_token_ids.pkl'), 'wb') as file:
        pickle.dump(qid_2_query_token_ids, file)
    
    with open(os.path.join(args.output_dir, 'pid_2_passage_token_ids.pkl'), 'wb') as file:
        pickle.dump(pid_2_passage_token_ids, file)  

    logger.info(f"done!")

def build_triplets(args):
    
    triplets = []
    qid_set = set()
    pid_set = set()
    with open(args.triplet_file) as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split('\t')
            triplets.append((qid, pos_id, neg_id))
            qid_set.add(qid)
            pid_set.add(pos_id)
            pid_set.add(neg_id)
            # only keep valid triplets
            # if qid in qid_2_query and pos_id in pid_2_passage and neg_id in pid_2_passage:

    random.shuffle(triplets)
    sub_sample = 1000000
    triplets = triplets[:sub_sample]

    logger.info(f"there are {len(triplets)} triplets in the training data")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'triplets.pkl'), 'wb') as file:
        pickle.dump(triplets, file)
    
    logger.info(f"done triplets!")

    return qid_set, pid_set

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--triplet_file', required=True)
    parser.add_argument('--passage_collection', required=True)
    parser.add_argument('--query_collection', required=True)
    parser.add_argument('--truncate', type=int, default=128)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--name_tag', type=str, default='')
    args = parser.parse_args()

    print(args)

    qid_set, pid_set = build_triplets(args)
    build_tokenized_data(args, qid_set, pid_set)
    