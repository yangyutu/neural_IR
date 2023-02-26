
msmarco_data_root=/mnt/d/MLData/data/msmarco_passage

python tasks/retrieval_hard_neg/prepare_hard_neg/build_passage_hard_triplet_crossencoder_train_data.py \
--triplet_file ${msmarco_data_root}/triplets/qidpidtriples.train.small.2.tsv \
--passage_collection ${msmarco_data_root}/collection.tsv \
--query_collection ${msmarco_data_root}/queries.train.tsv \
--query_embedding_path ./experiments/precomputed_embeddings/hard_neg_prepare/query_embed.pkl \
--qrel_path ${msmarco_data_root}/qrels.train.tsv \
--index_save_path ./experiments/index/hard_neg_prepare/ivf_nlist_1000.index \
--nprobe 30 \
--topk 200 \
--num_proc 24 \
--retrival_output_path ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed_hardneg/retrieval.json \
--output_dir ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed_hardneg