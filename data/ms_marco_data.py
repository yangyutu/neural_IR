import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer

class MSTripletData(Dataset):
    def __init__(self, triplet_path, pid_2_passage_token_ids_path, qid_2_query_token_ids_path, tokenizer_name, max_len=128):
        super().__init__()
        self.triplet_path = triplet_path
        self.pid_2_passage_token_ids_path = pid_2_passage_token_ids_path
        self.qid_2_query_token_ids_path = qid_2_query_token_ids_path
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)
        self.max_len = max_len
        self._load_data()

    def _load_data(self):
        
        with open(self.triplet_path, 'rb') as file:
            self.triplets = pickle.load(file)

        with open(self.pid_2_passage_token_ids_path, 'rb') as file:
            self.pid_2_passage_token_ids = pickle.load(file)

        with open(self.qid_2_query_token_ids_path, 'rb') as file:
            self.qid_2_query_token_ids = pickle.load(file)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index: int):

        qid, pos_id, neg_id = self.triplets[index]

        encoded_query = self.tokenizer.encode_plus(
            self.qid_2_query_token_ids[qid],
            return_tensors='pt',
            max_length=self.max_len, 
            truncation='longest_first', 
            padding='max_length')
        
        encoded_pos = self.tokenizer.encode_plus(
            self.pid_2_passage_token_ids[pos_id],
            return_tensors='pt',
            max_length=self.max_len, 
            truncation='longest_first', 
            padding='max_length')

        encoded_neg = self.tokenizer.encode_plus(
            self.pid_2_passage_token_ids[neg_id],
            return_tensors='pt',
            max_length=self.max_len, 
            truncation='longest_first', 
            padding='max_length')

        return encoded_query, encoded_pos, encoded_neg

def get_data_loader(dataset, batch_size=32, shuffle=True, num_workers=64):
    data_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

    return data_loader        
    

if __name__ == '__main__':

    pid_2_passage_token_ids_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/pid_2_passage_token_ids.pkl'
    qid_2_query_token_ids_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/qid_2_query_token_ids.pkl'
    triplet_path = '/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data/triplets.pkl'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)
    
    text = "Replace me by any text and words you'd like tokenization."
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    tokenizer.encode_plus(ids)
    
    
    dataset = MSTripletData(triplet_path, pid_2_passage_token_ids_path, qid_2_query_token_ids_path, 'distilbert-base-uncased', max_len=128)
    
    print(dataset[0])

    data_loader = get_data_loader(dataset, batch_size=4)

    print(next(iter(data_loader)))


