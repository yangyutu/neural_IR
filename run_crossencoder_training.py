import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer

from dataset.ms_marco_data import MSQDPairTrainDataModule, MSQDEvalDataModule
from models.cross_encoder_finetune import CrossEncoderFineTune


def run(args):
    wandb_logger = WandbLogger(
        project=args.project_name,  # group runs in "MNIST" project
        log_model="all",
        save_dir=args.default_root_dir,
        tags=[args.pretrained_model_name],
    )  # log all new checkpoints during training

    seed_everything(args.seed, workers=True)

    train_data_module = MSQDPairTrainDataModule(
        args.train_triplet_path,
        args.train_qid_2_query_path,
        args.train_pid_2_passage_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_data_module = MSQDEvalDataModule(
        args.val_qid_2_query_path,
        args.val_pid_2_passage_path,
        args.val_query_candidates_path,
        args.val_qrels_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    traindata_loader = train_data_module.train_dataloader()
    valdata_loader = val_data_module.test_dataloader()

    model = CrossEncoderFineTune(
        pretrained_model_name=args.pretrained_model_name,
        num_classes=2,
        truncate=args.max_len,
        lr=args.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        every_n_train_steps=args.model_save_every_n_steps,
        monitor="val_mrr",
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        precision=args.precision,
        val_check_interval=args.model_validate_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        callbacks=[
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate),
            checkpoint_callback,
            lr_monitor,
        ],
        logger=wandb_logger,
        deterministic=True,
    )

    # trainer.validate(model, valdata_loader)
    trainer.fit(
        model, train_dataloaders=traindata_loader, val_dataloaders=valdata_loader
    )
    # trainer.fit(model, train_dataloaders=traindata_loader)


def debug():
    model = CrossEncoderFineTune(
        pretrained_model_name="bert-base-uncased",
        num_classes=2,
        truncate=120,
    )

    data_root = "/mnt/d/MLData/Repos/neural_IR/experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_tiny"
    pid_2_passage_path = os.path.join(data_root, "pid_2_passage_text.pkl")
    qid_2_query_path = os.path.join(data_root, "qid_2_query_text.pkl")
    triplet_path = os.path.join(data_root, "triplets.pkl")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

    train_data_module = MSQDPairTrainDataModule(
        triplet_path,
        qid_2_query_path,
        pid_2_passage_path,
        batch_size=32,
    )
    traindata_loader = train_data_module.train_dataloader()

    for batch in traindata_loader:
        model.cuda()
        model.training_step(batch)
        break


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)
    parser.add_argument("--model_save_every_n_steps", type=int, default=25000)
    parser.add_argument("--model_validate_every_n_steps", type=int, default=25000)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)

    # model specific arguments
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)

    # dataset specific arguments
    parser.add_argument("--train_triplet_path", type=str, required=True)
    parser.add_argument("--train_pid_2_passage_path", type=str, required=True)
    parser.add_argument("--train_qid_2_query_path", type=str, required=True)
    parser.add_argument("--val_pid_2_passage_path", type=str, required=True)
    parser.add_argument("--val_qid_2_query_path", type=str, required=True)
    parser.add_argument("--val_query_candidates_path", type=str, required=True)
    parser.add_argument("--val_qrels_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # debug()
    args = parse_arguments()
    run(args)
#    print(vars(args))
#    run(args)
