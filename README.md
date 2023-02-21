# neural_IR

This repository contains a PyTorch framework for training neural based information retrieval applications.

### Dependencies
```
pip install -r requirements.txt
```

## (Update 2023) Experiments for MS MARCO passage reranking task

Re-ranking takes top 1000 candidates from BM25 as the input.

### Training




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

## Experiments for MS MARCO passage reranking task

### Data preparation

1) Download from https://microsoft.github.io/msmarco/Datasets

2) Prepare triplet training data

We don't need the full triplet set idpidtriples.train.full.2.tsv for the model to converge. We can sample 3% from it. 

Use /mnt/d/MLData/data/msmarco_passage/data_explorer.ipynb to sample the 

qidpidtriples.train.medium_mixed.2.tsv

Then run the following 

```
python data_helper/msmarco/build_passage_triplet_crossencoder_train_data.py \
--triplet_file /mnt/d/MLData/data/msmarco_passage/triplets/qidpidtriples.train.medium_mixed.2.tsv \
--passage_collection /mnt/d/MLData/data/msmarco_passage/collection.tsv \
--query_collection /mnt/d/MLData/data/msmarco_passage/queries/queries.train.tsv \
--output_dir ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed
```
3) Prepare dev data for validation, which has 6980 queries (each query has around ~1000 candidates)

```
python data_helper/msmarco/build_passage_ranking_dev_data.py \
--passage_collection /mnt/d/MLData/data/msmarco_passage/collection.tsv \
--query_collection /mnt/d/MLData/data/msmarco_passage/queries.dev.small.tsv \
--query_candidates_path ./assets/msmarco/query_2_top_1000_passage_BM25.json \
--output_dir experiments/msmarco_psg_ranking/dev_data_sz_1000 \
--sample_size 1000
```

### Model training and visualization

We use pytorch-lightning to support distributed model training and monitoring.

```
python run_crossencoder_training.py \
--gpus 1 \
--limit_train_batches 1.0 \
--max_epochs 3 \
--model_save_every_n_steps 20000 \
--model_validate_every_n_steps 5000 \
--lr 3e-6 \
--batch_size 128 \
--num_workers 24 \
--pretrained_model_name microsoft/MiniLM-L12-H384-uncased \
--train_triplet_path ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed/triplets.pkl \ --train_pid_2_passage_path ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed/ pid_2_passage_text.pkl \
--train_qid_2_query_path ./experiments/msmarco_psg_rankingcross_encoder_triplet_train_data_medium_mixed/qid_2_query_text.pkl \
--val_pid_2_passage_path ./experiments/msmarco_psg_ranking/dev_data_sz_500/pid_2_passage_text.pkl \
--val_qid_2_query_path ./experiments/msmarco_psg_ranking/dev_data_sz_500/qid_2_query_text.pkl \
--val_query_candidates_path ./experiments/msmarco_psg_ranking/dev_data_sz_500/qid_2_top_1000_passage_BM25_subset.json \ 
--val_qrels_path ./assets/msmarco/query_2_groundtruth_passage_small.json \
--max_len 128 \
--project_name crossencoder_rerank_MSMARCO_bert \
--default_root_dir ./experiments/msmarco_psg_ranking/logs
```

Visualization using Tensorboard

> tensorboard --logdir mylogdir


### End-to-end retrieval

To enable efficient end-to-end dense retrieval, we can precompute and index passage embeddings offline. We use IVFPQ indexing strategy from FAISS. The pre-computation and indexing can be executed via the following

```
python run_retrieval_indexing.py \
--pid_2_passage_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full/pid_2_passage_token_ids.pkl \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--embedding_save_path passage_embedding.pkl \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/version_14/checkpoints/epoch=0-step=156249.ckpt \
--num_partitions 2000 \
--subquantizer_number 8 \
--subquantizer_codebook_size 8 \
--model_name bert_encoder \
--loss_name triplet_loss \
--index_save_path embed.index
```

Given a list of query, the end-to-end retrieval of relevant passages from the 8.8M passages can be run via

```
python run_retrieval.py \
--qid_2_query_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data/qid_2_query_token_ids.pkl \
--pid_2_embedding_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/passage_embedding.pkl \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/version_12/checkpoints/epoch=1-step=312499.ckpt \
--index_save_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/embed.index \
--number_nearest_neighbors 5 \
--output_file retrieval.tsv \
--model_name bert_encoder \
--loss_name triplet_loss
```






### Notebooks for analyzing ranking results

notebooks/Result_analysis.ipynb


## Data size stats

### query and passages

collection.tsv has 8,841,823 passages. One example is

queries.train.tsv has 808,731 (qid, query_text) rows 

qrels.train.tsv has 532761 (qid, iter, pid, label) rows for positive passages associated with each qid. Among them, we have 502939 distinct qids.

queries_dev.tsv has 101,093 (qid, query_text) rows 

qrels.dev.tsv has 59273 (qid, iter, pid, label) rows for positive passages associated with each qid

### Triplet training data

qidpidtriples.train.full.2.tsv has 397,768,673 rows

It has 400782 distinct qids, which is a subset of 502939 qids

Note that to train a cross-encoder to reranking, we typically only need a small fraction of qidpidtriples.train.full.2.tsv.

For example, 
qidpidtriples.train.medium_mixed.2.tsv has 11933060 rows (3% of the full training triplets), 395465 distinct qids, and 5464050 distinct pid