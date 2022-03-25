# neural_IR

This repository contains a PyTorch framework for training neural based information retrieval applications.

### Dependencies
```
pip install -r requirements.txt
```


## Experiments for MS MARCO passage retrieval task

### Data preparation

1) Download from https://microsoft.github.io/msmarco/Datasets

2) Prepare triplet training data
```
python data_helepr/msmarco/build_passage_triplet_train_data.py \
--tokenizer_name distilbert-base-uncased \
--triplet_file /home/ubuntu/MLData/work/Repos/NeuralIR/data/qidpidtriples.train.full.2.tsv \
--passage_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/collection.tsv \
--query_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/queries.train.tsv \
--truncate 128 \
--output_dir /home/ubuntu/MLData/work/Repos/NeuralIR/BERTEncoder/experiments/msmarco_psg/train_data
```
3) Prepare dev data for validation
```
python build_passage_ranking_dev_data.py \
--tokenizer_name distilbert-base-uncased \
--query_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/queries.dev.small.tsv \
--truncate 128 \
--output_dir /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data
```

### Model training and visualization

We use pytorch-lightning to support distributed model training and monitoring.

```
data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full_10M/
export CUDA_VISIBLE_DEVICES="5"
python run_task.py \
--gpus 1 \
--limit_train_batches 1.0 \
--max_epochs 30 \
--lr 1e-6 \
--batch_size 64 \
--margin 15.0 \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_name bert_encoder \
--loss_name triplet_loss \
--triplet_path ${data_root}triplets.pkl \
--pid_2_passage_token_ids_path ${data_root}pid_2_passage_token_ids.pkl \
--qid_2_query_token_ids_path ${data_root}qid_2_query_token_ids.pkl \
--max_len 128 \
--default_root_dir /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments
```

Visualization using Tensorboard

> tensorboard --logdir mylogdir


### End-to-end retrieval

To enable 


### Re-rank 

Re-ranking takes top 1000 candidates from BM25 as the input.

```
python run_reranking.py \
--qid_2_query_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data/qid_2_query_token_ids.pkl \
--pid_2_passage_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full/pid_2_passage_token_ids.pkl \
--re_rank_input_file_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/assets/msmarco/query_2_top_1000_passage_BM25.json \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/version_12/checkpoints/epoch=1-step=312499.ckpt \
--model_name bert_encoder \
--loss_name triplet_loss \
--output_file ranking.tsv
```

### Notebooks for analyzing ranking results

notebooks/Result_analysis.ipynb






