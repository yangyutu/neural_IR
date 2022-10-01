import json
import os
import pickle
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def text_pair_collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    text_pairs, labels = zip(*data)

    return list(text_pairs), torch.tensor(labels).long()


def text_pair_collate_fn_with_qd_id(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    qd_ids, text_pairs, labels = zip(*data)

    return list(qd_ids), list(text_pairs), torch.tensor(labels).long()


class MSQDPairTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        triplet_path: str,
        query_path: str,
        passage_path: str,
        batch_size: int = 64,
        num_workers: int = 8,
        collate_fn=text_pair_collate_fn,
    ):
        super().__init__()
        self.triplet_path = triplet_path
        self.query_path = query_path
        self.passage_path = passage_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.ms_qd_pair_data_train = MSQDPairData(
            self.triplet_path, self.query_path, self.passage_path
        )

    def train_dataloader(self):
        return DataLoader(
            self.ms_qd_pair_data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=text_pair_collate_fn,
        )


class MSQDEvalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        query_path: str,
        passage_path: str,
        query_candidates_path: str,
        qrels_path: str,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.query_candidates_path = query_candidates_path
        self.query_path = query_path
        self.passage_path = passage_path
        self.qrels_path = qrels_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.ms_qd_rank_data_val = MSQDRankEvalData(
            self.query_candidates_path,
            self.query_path,
            self.passage_path,
            self.qrels_path,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ms_qd_rank_data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=text_pair_collate_fn_with_qd_id,
        )


class MSQDPairData(Dataset):
    def __init__(self, triplet_path: str, query_path: str, passage_path: str):
        super().__init__()
        self.triplet_path = triplet_path
        self.query_path = query_path
        self.passage_path = passage_path
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        # self.max_len = max_len
        self._load_data()

    def _load_data(self):

        with open(self.triplet_path, "rb") as file:
            triplets = pickle.load(file)

        with open(self.query_path, "rb") as file:
            self.qid_2_query = pickle.load(file)

        with open(self.passage_path, "rb") as file:
            self.pid_2_passage = pickle.load(file)

        self.pairs = []
        for qid, pos_pid, neg_pid in triplets:
            self.pairs.append((qid, pos_pid, 1))
            self.pairs.append((qid, neg_pid, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):

        qid, pid, label = self.pairs[index]
        qid_text = self.qid_2_query[qid]
        pid_text = self.pid_2_passage[pid]

        return [qid_text, pid_text], label


class MSQDRankEvalData(Dataset):
    def __init__(
        self,
        query_candidates_path: str,
        query_path: str,
        passage_path: str,
        qrels_path: str,
    ):
        super().__init__()
        self.query_candidates_path = query_candidates_path
        self.query_path = query_path
        self.passage_path = passage_path
        self.qrels_path = qrels_path
        self._load_data()

    def _load_data(self):

        with open(self.query_candidates_path, "r") as file:
            self.query_candidates = json.load(file)

        with open(self.query_path, "rb") as file:
            self.qid_2_query = pickle.load(file)

        with open(self.passage_path, "rb") as file:
            self.pid_2_passage = pickle.load(file)

        with open(self.qrels_path, "r") as file:
            self.qid_2_relevant_pid = json.load(file)

        self.pairs = []
        for qid, candidate_list in self.query_candidates.items():
            for pid in candidate_list:
                # if pid is a relevant passage
                if pid in self.qid_2_relevant_pid[qid]:
                    self.pairs.append((qid, pid, 1))
                else:
                    self.pairs.append((qid, pid, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):

        qid, pid, label = self.pairs[index]
        qid_text = self.qid_2_query[qid]
        pid_text = self.pid_2_passage[pid]

        return [qid, pid, label], [qid_text, pid_text], label


class MSTripletData(Dataset):
    def __init__(
        self,
        triplet_path: str,
        pid_2_passage_token_ids_path: str,
        qid_2_query_token_ids_path: str,
        tokenizer_name: str,
        max_len: int = 128,
    ):
        super().__init__()
        self.triplet_path = triplet_path
        self.pid_2_passage_token_ids_path = pid_2_passage_token_ids_path
        self.qid_2_query_token_ids_path = qid_2_query_token_ids_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_len = max_len
        self._load_data()

    def _load_data(self):

        with open(self.triplet_path, "rb") as file:
            self.triplets = pickle.load(file)

        with open(self.pid_2_passage_token_ids_path, "rb") as file:
            self.pid_2_passage_token_ids = pickle.load(file)

        with open(self.qid_2_query_token_ids_path, "rb") as file:
            self.qid_2_query_token_ids = pickle.load(file)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index: int):

        qid, pos_id, neg_id = self.triplets[index]

        encoded_query = self.tokenizer.encode_plus(
            self.qid_2_query_token_ids[qid],
            return_tensors="pt",
            max_length=self.max_len,
            truncation="longest_first",
            padding="max_length",
        )

        encoded_pos = self.tokenizer.encode_plus(
            self.pid_2_passage_token_ids[pos_id],
            return_tensors="pt",
            max_length=self.max_len,
            truncation="longest_first",
            padding="max_length",
        )

        encoded_neg = self.tokenizer.encode_plus(
            self.pid_2_passage_token_ids[neg_id],
            return_tensors="pt",
            max_length=self.max_len,
            truncation="longest_first",
            padding="max_length",
        )

        return encoded_query, encoded_pos, encoded_neg

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MSTripletData")
        parser.add_argument("--triplet_path", type=str, required=True)
        parser.add_argument("--pid_2_passage_token_ids_path", type=str, required=True)
        parser.add_argument("--qid_2_query_token_ids_path", type=str, required=True)
        parser.add_argument("--tokenizer_name", type=str, required=True)
        parser.add_argument("--max_len", type=int, default=128)
        return parent_parser


class MSTextData(Dataset):
    def __init__(
        self, id_2_token_ids_path: str, tokenizer_name: str, max_len: int = 128
    ):
        super().__init__()
        self.id_2_token_ids_path = id_2_token_ids_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_len = max_len
        self._load_data()

    def _load_data(self):

        with open(self.id_2_token_ids_path, "rb") as file:
            self.id_2_token_ids = pickle.load(file)

        self.id_2_token_ids_tuple = list(
            [(k, v) for k, v in self.id_2_token_ids.items()]
        )

    def __len__(self):
        return len(self.id_2_token_ids_tuple)

    def __getitem__(self, index: int):

        encoded_text = self.tokenizer.encode_plus(
            self.id_2_token_ids_tuple[index][1],
            return_tensors="pt",
            max_length=self.max_len,
            truncation="longest_first",
            padding="max_length",
        )

        return self.id_2_token_ids_tuple[index][0], encoded_text


def get_data_loader(
    dataset, batch_size=32, shuffle=True, num_workers=64, collate_fn=None
):
    if collate_fn:
        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
    else:
        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    return data_loader


if __name__ == "__main__":

    # pid_2_passage_token_ids_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/pid_2_passage_token_ids.pkl'
    # qid_2_query_token_ids_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/qid_2_query_token_ids.pkl'
    # triplet_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/triplets.pkl'
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)

    # text = "Replace me by any text and words you'd like tokenization."
    # ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    # tokenizer.encode_plus(ids)

    # dataset = MSTripletData(triplet_path, pid_2_passage_token_ids_path, qid_2_query_token_ids_path, 'distilbert-base-uncased', max_len=128)

    # print(dataset[0])

    # data_loader = get_data_loader(dataset, batch_size=4)

    # print(next(iter(data_loader)))

    data_root = "/mnt/d/MLData/Repos/neural_IR/experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_tiny"
    pid_2_passages_path = os.path.join(data_root, "pid_2_passage_text.pkl")
    qid_2_query_path = os.path.join(data_root, "qid_2_query_text.pkl")
    triplet_path = os.path.join(data_root, "triplets.pkl")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)

    dataset = MSQDPairData(triplet_path, qid_2_query_path, pid_2_passages_path)

    print(dataset[0])

    data_loader = get_data_loader(
        dataset, batch_size=4, collate_fn=text_pair_collate_fn
    )

    for batch in data_loader:
        text_pairs, label = batch
        encoded = tokenizer(text_pairs)
        break
