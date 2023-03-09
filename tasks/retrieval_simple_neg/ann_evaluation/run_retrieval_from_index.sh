query_embedding_path=./experiments/precomputed_embeddings/simple_neg/query_embed.pkl
index_save_path=./experiments/index/simple_neg/exact_embed.index
retrieval_result_path=./experiments/retrieval_results/simple_neg/out.json

python tasks/retrieval/ann_evaluation/run_retrieval_from_index.py \
--query_embedding_path ${query_embedding_path} \
--index_save_path ${index_save_path} \
--output_path ${retrieval_result_path}