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
python build_passage_triplet_train_data.py \
--tokenizer_name distilbert-base-uncased \
--triplet_file /home/ubuntu/MLData/work/Repos/NeuralIR/data/qidpidtriples.train.full.2.tsv \
--passage_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/collection.tsv \
--query_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/queries.train.tsv \
--truncate 128 \
--output_dir /home/ubuntu/MLData/work/Repos/NeuralIR/BERTEncoder/experiments/msmarco_psg/train_data \
--name_tag full
```
### Model training


Visualization using Tensorboard

> tensorboard --logdir mylogdir

### Model validation


### End-to-end retrieval


### Retrieval evaluation






