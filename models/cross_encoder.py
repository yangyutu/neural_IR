import collections

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel, AutoTokenizer
from torchmetrics import RetrievalMRR, RetrievalRecall


class BertCrossEncoder(pl.LightningModule):
    def __init__(
        self, pretrained_model_name, num_classes, truncate, lr=1e-6, warm_up_step=10000
    ):
        super().__init__()
        self.cross_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.linear = nn.Linear(self.cross_encoder.config.hidden_size, num_classes)
        self.truncate = truncate
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr
        self.warm_up_step = warm_up_step

    def model_forward(self, input_text_pairs):

        encoded_inputs = self.tokenizer(
            input_text_pairs,
            return_tensors="pt",
            max_length=self.truncate,
            truncation="longest_first",
            padding="max_length",
        )
        encoder_outputs = self.cross_encoder(
            input_ids=encoded_inputs["input_ids"].cuda(),
            attention_mask=encoded_inputs["attention_mask"].cuda(),
        )
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # with linear rate warm up
    # https://github.com/Lightning-AI/lightning/issues/328
    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
    #     if self.trainer.global_step < self.warm_up_step:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warm_up_step)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.lr

    #     optimizer.step()
    #     optimizer.zero_grad()

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    #     def lr_foo(epoch):
    #         if epoch < self.warm_up_step:
    #             # warm up lr
    #             lr_scale = 0.1 ** (self.warm_up_step - epoch)
    #         else:
    #             lr_scale = 0.95 ** epoch

    #         return lr_scale

    #     scheduler = LambdaLR(
    #         optimizer,
    #         lr_lambda=lr_foo
    #     )

    #     return [optimizer], [scheduler]


if __name__ == "__main__":
    model = BertCrossEncoder(
        pretrained_model_name="distilbert-base-uncased",
        hidden_dim=768,
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
