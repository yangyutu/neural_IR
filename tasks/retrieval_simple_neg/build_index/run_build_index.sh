doc_embedding_path=./experiments/precomputed_embeddings/simple_neg/doc_embed.pkl
index_save_path=./experiments/index/simple_neg/exact_embed.index

python tasks/retrieval/build_index/run_build_index.py \
--doc_embedding_path ${doc_embedding_path} \
--index_save_path ${index_save_path} \
--index_type exact


