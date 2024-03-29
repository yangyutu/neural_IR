import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from models.cross_encoder_finetune import CrossEncoderFineTune
from dataset.ms_marco_data import MSQDEvalDataModule


def run(args):

    wandb_logger = WandbLogger(
        project=args.project_name,  # group runs in "MNIST" project
        log_model="all",
        save_dir=args.default_root_dir,
    )

    data_module = MSQDEvalDataModule(
        args.qid_2_query_path,
        args.pid_2_passage_path,
        args.query_candidates_path,
        args.qrels_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader = data_module.test_dataloader()

    model = CrossEncoderFineTune(
        pretrained_model_name=args.pretrained_model_name,
        num_classes=2,
        truncate=args.max_len,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=args.default_root_dir,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=wandb_logger,
    )

    trainer.validate(model, data_loader, ckpt_path=args.model_checkpoint)


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)

    # model specific arguments
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    # dataset specific arguments
    parser.add_argument("--query_candidates_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    parser.add_argument("--pid_2_passage_path", type=str, required=True)
    parser.add_argument("--qid_2_query_path", type=str, required=True)
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
