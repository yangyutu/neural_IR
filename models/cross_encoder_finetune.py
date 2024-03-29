import collections
from typing import Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel, AutoTokenizer
from torchmetrics import RetrievalMRR, RetrievalRecall
from models.utils import batch_to_device


class CrossEncoderFineTune(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name,
        truncate,
        num_classes=2,
        config: Dict = [],
    ):
        super().__init__()
        self.cross_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.linear = nn.Linear(self.cross_encoder.config.hidden_size, num_classes)
        self.truncate = truncate
        self.loss_func = nn.CrossEntropyLoss()
        self.config = config
        self.save_hyperparameters()

    def model_forward(self, input_text_pairs):

        encoded_inputs = self.tokenizer(
            input_text_pairs,
            return_tensors="pt",
            max_length=self.truncate,
            truncation="longest_first",
            padding="max_length",
        )
        # encoded_inputs is a dictionary with three keys: input_ids, attention_mask, and token_type_ids
        # the position ids [0, 1, ..., seq_len - 1] will be generated by default on the fly in the cross_encoder
        encoded_inputs = batch_to_device(encoded_inputs, target_device=self.device)
        encoder_outputs = self.cross_encoder(**encoded_inputs)
        # encoder_outputs have two elements: last_hidden_state (shape: batch_size x seq_len x hidden_dim) and pooler_output (shape: batch_size x hidden_dim)
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # bert_output is the last layer's hidden state
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_representations = last_hidden_states[:, 0, :]
        predictions = self.linear(cls_representations)

        return predictions

    def training_step(self, batch, batch_idx=0):

        input_text_pairs, labels = batch
        predictions = self.model_forward(input_text_pairs)
        loss = self.loss_func(predictions, labels)

        self.log("loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:

        self.query_candidates_scores = collections.defaultdict(list)
        self.query_ids = []
        self.preds = []
        self.rel_label = []

    def validation_step(self, batch, batch_idx=0):
        qd_ids, input_text_pairs, labels = batch
        predictions = self.model_forward(input_text_pairs)

        for qd_id, pred in zip(qd_ids, predictions):
            self.query_candidates_scores[qd_id[0]].append((qd_id[2], pred))
            self.query_ids.append(int(qd_id[0]))
            self.preds.append(pred[1].item())
            self.rel_label.append(qd_id[2])

    def validation_epoch_end(self, output_results):

        preds = torch.Tensor(self.preds)
        targets = torch.tensor(self.rel_label).long()
        indexes = torch.tensor(self.query_ids).long()
        recall_at_5 = RetrievalRecall(k=5)(preds, targets, indexes=indexes)
        recall_at_20 = RetrievalRecall(k=20)(preds, targets, indexes=indexes)
        recall_at_100 = RetrievalRecall(k=100)(preds, targets, indexes=indexes)
        recall_at_200 = RetrievalRecall(k=100)(preds, targets, indexes=indexes)
        recall_at_1000 = RetrievalRecall(k=1000)(preds, targets, indexes=indexes)

        mrr = RetrievalMRR()(preds, targets, indexes=indexes)

        self.log("val_recall@5", recall_at_5, prog_bar=True)
        self.log("val_recall@20", recall_at_20, prog_bar=True)
        self.log("val_recall@100", recall_at_100, prog_bar=True)
        self.log("val_recall@200", recall_at_200, prog_bar=True)
        self.log("val_recall@1000", recall_at_1000, prog_bar=True)
        self.log("val_mrr", mrr, on_epoch=True, prog_bar=True)

        return mrr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-6))
        return optimizer


if __name__ == "__main__":
    model = CrossEncoderFineTune(
        pretrained_model_name="distilbert-base-uncased",
        num_classes=2,
        truncate=120,
    )

    import os

    from dataset.ms_marco_data import (
        MSQDPairData,
        get_data_loader,
        text_pair_collate_fn,
    )

    data_root = "/mnt/d/MLData/Repos/neural_IR/experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_tiny"
    pid_2_passages_path = os.path.join(data_root, "pid_2_passage_text.pkl")
    qid_2_query_path = os.path.join(data_root, "qid_2_query_text.pkl")
    triplet_path = os.path.join(data_root, "triplets.pkl")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)

    dataset = MSQDPairData(triplet_path, qid_2_query_path, pid_2_passages_path)

    data_loader = get_data_loader(
        dataset, batch_size=4, collate_fn=text_pair_collate_fn
    )

    for batch in data_loader:
        model.training_step(batch)
        break
