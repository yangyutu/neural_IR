time python tasks/retrieval/ann_evaluation/run_retrieval_from_index.py \
--query_embedding_path ./experiments/precomputed_embeddings/query_embed.pkl \
--index_save_path ./experiments/index/faiss_compare/exact_embed.index \
--output_path ./experiments/retrieval_results/faiss_compare/exact_out.json

# time python tasks/retrieval/ann_evaluation/run_retrieval_from_index.py \
# --query_embedding_path ./experiments/precomputed_embeddings/query_embed.pkl \
# --index_save_path ./experiments/index/faiss_compare/ivf_nlist_1000.index \
# --output_path ./experiments/retrieval_results/faiss_compare/ivf_out.json \
# --nprobe 100

# time python tasks/retrieval/ann_evaluation/run_retrieval_from_index.py \
# --query_embedding_path ./experiments/precomputed_embeddings/query_embed.pkl \
# --index_save_path ./experiments/index/faiss_compare/ivfpq_nlist_1000_subnum_16_subcbsize_8.index \
# --output_path ./experiments/retrieval_results/faiss_compare/ivfpq_out.json \
# --nprobe 100