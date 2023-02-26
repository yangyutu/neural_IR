
python tasks/retrieval_hard_neg/prepare_hard_neg/build_passage_hard_triplet_crossencoder_train_data.py \
--triplet_file /mnt/d/MLData/data/msmarco_passage/triplets/qidpidtriples.train.tiny.2.tsv \
--passage_collection /mnt/d/MLData/data/msmarco_passage/collection.tsv \
--query_collection /mnt/d/MLData/data/msmarco_passage/queries.train.tsv \
--query_embedding_path ./experiments/precomputed_embeddings/hard_neg_prepare/query_embed.pkl \
--qrel_path /mnt/d/MLData/data/msmarco_passage/qrels.train.tsv \
--index_save_path ./experiments/index/hard_neg_prepare/exact_embed.index \
--topk 200 \
--num_proc 24 \
--retrival_output_path ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed_hardneg/retrieval.json \
--output_dir ./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed_hardneg