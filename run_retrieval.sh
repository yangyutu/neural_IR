python run_retrieval.py \
--id_2_query_embed_filename /mnt/d/MLData/Repos/neural_IR/precomputed_embeddings/query_embed.pkl \
--id_2_doc_embed_filename /mnt/d/MLData/Repos/neural_IR/precomputed_embeddings/doc_embed.pkl \
--topk 1000 \
--output_rank_file_path ./retrieval_rank.tsv