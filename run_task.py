import sys, os

from fastprogress import progress_bar
#sys.path.append("/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR")
from data.ms_marco_data import MSTripletData, get_data_loader
from models.model_interface import PLModelTripletInterface
from models.bert_encoder import BertEncoder
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

def run(args):
    data_root = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_tiny/'
    pid_2_passage_token_ids_path = os.path.join(data_root, 'pid_2_passage_token_ids.pkl')
    qid_2_query_token_ids_path = os.path.join(data_root, 'qid_2_query_token_ids.pkl')
    tokenizer_name = 'distilbert-base-uncased'
    default_root_dir = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments'
    triplet_path = os.path.join(data_root, 'triplets.pkl')
    dataset = MSTripletData(args.triplet_path, args.pid_2_passage_token_ids_path, args.qid_2_query_token_ids_path, args.tokenizer_name, max_len=args.max_len)
    train_loader = get_data_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    #model = PLModelTripletInterface(model_name='bert_encoder', loss_name='triplet_loss', lr=1e-6, lr_scheduler=None, pretrained_model_name="distilbert-base-uncased", margin=1.0)

    model = PLModelTripletInterface(**vars(args))

    # training
    # trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.1, max_epochs=3, progress_bar_refresh_rate=20, default_root_dir=default_root_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    trainer = pl.Trainer(gpus=args.gpus, \
                        precision=args.precision,
                        limit_train_batches=args.limit_train_batches,
                        max_epochs=args.max_epochs,
                        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                        default_root_dir=args.default_root_dir,
                        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)


    PLModelTripletInterface.add_model_specific_args(parser)
    BertEncoder.add_model_specific_args(parser)
    MSTripletData.add_data_specific_args(parser)
    args = parser.parse_args()
    
    print(vars(args))
    run(args)