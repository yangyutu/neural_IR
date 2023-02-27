import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from models.cross_encoder_finetune import CrossEncoderFineTune
from dataset.ms_marco_data import MSQDEvalDataModule, MSQDPairTrainDataModule


def run(args):

    data_module = MSQDPairTrainDataModule(
        args.train_triplet_path,
        args.train_qid_2_query_path,
        args.train_pid_2_passage_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loader = data_module.train_dataloader()

    model = CrossEncoderFineTune.load_from_checkpoint(args.model_ckpt)
    
    


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
