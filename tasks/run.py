import sys, os
#sys.path.append("/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR")
from data.ms_marco_data import MSTripletData, get_data_loader
from models.model_interface import PLModelTripletInterface
from transformers import AutoTokenizer
import pytorch_lightning as pl

def run():
    data_root = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_tiny/'
    pid_2_passage_token_ids_path = os.path.join(data_root, 'pid_2_passage_token_ids.pkl')
    qid_2_query_token_ids_path = os.path.join(data_root, 'qid_2_query_token_ids.pkl')
    triplet_path = os.path.join(data_root, 'triplets.pkl')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MSTripletData(triplet_path, pid_2_passage_token_ids_path, qid_2_query_token_ids_path, tokenizer, max_len=128)

    train_loader = get_data_loader(dataset, batch_size=32, num_workers=16)

    model = PLModelTripletInterface(model_name='bert_encoder', loss_name='triplet_loss', lr=1e-6, lr_scheduler=None, pretrained_model_name="distilbert-base-uncased", margin=1.0)

    # training
    trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.1)
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    run()