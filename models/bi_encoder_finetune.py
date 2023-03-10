import collections
from typing import Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import RetrievalMRR, RetrievalRecall
from transformers import AutoModel, AutoTokenizer
from losses.multiple_negative_ranking_loss import MultipleNegativesRankingLoss
from losses.mse_margin_distill_loss import MSEMarginDistillLoss
from losses.kl_divergence_distill_loss import KLDivergenceDistillLoss
from losses.listnet_distill_loss import ListNetDistillLoss



class BiEncoderFineTuneBase(pl.LightningModule):
    def __init__(
        self,
        query_encoder: nn.Module,
        doc_encoder: nn.Module,
        config: Dict = [],
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.config = config
        self.initialize_loss()
        self.save_hyperparameters()

    def initialize_loss(self):

        raise NotImplementedError()

    def compute_query_embeddings(self, input_text_list):
        return self.query_encoder.encode(
            input_text_list, device=self.device, token_type_id=0
        )

    def compute_doc_embeddings(self, input_text_list):
        return self.query_encoder.encode(
            input_text_list, device=self.device, token_type_id=1
        )

    def training_step(self, batch, batch_idx=0):

        raise NotImplementedError()

    def on_validation_epoch_start(self) -> None:

        self.query_candidates_scores = collections.defaultdict(list)
        self.query_ids = []
        self.preds = []
        self.rel_label = []

    def validation_step(self, batch, batch_idx=0):
        qd_ids, input_text_pairs, labels = batch
        query_text_list, doc_text_list = zip(*input_text_pairs)
        query_embeddings = self.query_encoder.encode(
            list(query_text_list), device=self.device, token_type_id=0
        )
        doc_embeddings = self.doc_encoder.encode(
            list(doc_text_list), device=self.device, token_type_id=1
        )
        similarities = torch.sum(query_embeddings * doc_embeddings, dim=1)

        for qd_id, sim_score in zip(qd_ids, similarities):
            self.query_candidates_scores[qd_id[0]].append((qd_id[2], sim_score))
            self.query_ids.append(int(qd_id[0]))
            self.preds.append(sim_score.item())
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


class BiEncoderFineTuneContrastLoss(BiEncoderFineTuneBase):
    def __init__(
        self,
        query_encoder: nn.Module,
        doc_encoder: nn.Module,
        config: Dict = [],
    ):
        super().__init__(
            query_encoder=query_encoder, doc_encoder=doc_encoder, config=config
        )

    def initialize_loss(self):
        self.loss_func = MultipleNegativesRankingLoss()

    def training_step(self, batch, batch_idx=0):

        query_text_list, pos_doc_text_list, neg_doc_text_list = batch
        # for query, we use token type id 0; for doc, we use token type id 1
        query_embeddings = self.query_encoder.encode(
            query_text_list, device=self.device, token_type_id=0
        )
        pos_doc_embeddings = self.doc_encoder.encode(
            pos_doc_text_list, device=self.device, token_type_id=1
        )
        neg_doc_embeddings = self.doc_encoder.encode(
            neg_doc_text_list, device=self.device, token_type_id=1
        )

        all_embeddings = [query_embeddings, pos_doc_embeddings, neg_doc_embeddings]

        loss = self.loss_func(all_embeddings)

        self.log(
            "loss",
            loss.item(),
            batch_size=len(query_text_list),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


class BiEncoderFineTuneContrastLossWithDistillation(BiEncoderFineTuneBase):
    def __init__(
        self,
        query_encoder: nn.Module,
        doc_encoder: nn.Module,
        teacher_query_encoder: nn.Module,
        teacher_doc_encoder: nn.Module,
        config: Dict = [],
    ):
        super().__init__(
            query_encoder=query_encoder, doc_encoder=doc_encoder, config=config
        )
        self.teacher_query_encoder = teacher_query_encoder
        self.teacher_doc_encoder = teacher_doc_encoder

    def initialize_loss(self):
        self.contrast_loss_func = MultipleNegativesRankingLoss()
        
        if self.config['distill_loss_type'] == 'mse_margin':
            self.distill_loss_func = MSEMarginDistillLoss()
        elif self.config['distill_loss_type'] == 'kl_div_loss':
            self.distill_loss_func = KLDivergenceDistillLoss()
        elif self.config['distill_loss_type'] == 'listnet_loss':
            self.distill_loss_func = ListNetDistillLoss()

    def training_step(self, batch, batch_idx=0):

        query_text_list, pos_doc_text_list, neg_doc_text_list = batch
        # for query, we use token type id 0; for doc, we use token type id 1
        query_embeddings = self.query_encoder.encode(
            query_text_list, device=self.device, token_type_id=0
        )
        pos_doc_embeddings = self.doc_encoder.encode(
            pos_doc_text_list, device=self.device, token_type_id=1
        )
        neg_doc_embeddings = self.doc_encoder.encode(
            neg_doc_text_list, device=self.device, token_type_id=1
        )

        all_embeddings = [query_embeddings, pos_doc_embeddings, neg_doc_embeddings]

        query_embeddings_teacher = self.teacher_query_encoder.encode(
            query_text_list, device=self.device, token_type_id=0
        )
        pos_doc_embeddings_teacher = self.teacher_doc_encoder.encode(
            pos_doc_text_list, device=self.device, token_type_id=1
        )
        neg_doc_embeddings_teacher = self.teacher_doc_encoder.encode(
            neg_doc_text_list, device=self.device, token_type_id=1
        )

        all_embeddings_teacher = [
            query_embeddings_teacher,
            pos_doc_embeddings_teacher,
            neg_doc_embeddings_teacher,
        ]

        loss_contrast = self.contrast_loss_func(all_embeddings)
        loss_dl = self.mse_margin_loss_func(all_embeddings, all_embeddings_teacher)

        loss = loss_contrast + self.config["distill_loss_coeff"] * loss_dl

        self.log_dict(
            {
                "loss": loss.item(),
                "loss_contrast": loss_contrast.item(),
                "loss_dl": loss_dl.item(),
            },
            batch_size=len(query_text_list),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


def _test_biencoder_contrast():
    model = BiEncoderFineTuneContrastLoss(
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


if __name__ == "__main__":
    _test_biencoder_contrast()
