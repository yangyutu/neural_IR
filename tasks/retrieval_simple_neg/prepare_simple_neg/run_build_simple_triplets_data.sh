
msmarco_data_root=/mnt/d/MLData/data/msmarco_passage

python tasks/retrieval_simple_neg/prepare_simple_neg/build_passage_simple_triplet_crossencoder_train_data.py \
--triplet_file ${msmarco_data_root}/triplets/qidpidtriples.train.medium_mixed.2.tsv \
--passage_collection ${msmarco_data_root}/collection.tsv \
--query_collection ${msmarco_data_root}/queries.train.tsv \
--qrel_path ${msmarco_data_root}/qrels.train.tsv \
--topk 200 \
--output_dir ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed_simpleneg