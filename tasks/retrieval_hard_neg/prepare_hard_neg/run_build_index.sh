doc_embed_path=./experiments/precomputed_embeddings/hard_neg_prepare/doc_embed.pkl

# python tasks/retrieval/build_index/run_build_index.py \
# --doc_embedding_path ${doc_embed_path} \
# --index_save_path ./experiments/index/hard_neg_prepare/exact_embed.index 

# ivf
nlist=1000
python tasks/retrieval/build_index/run_build_index.py \
--doc_embedding_path ${doc_embed_path} \
--index_type ivf \
--data_frac_for_training 0.1 \
--nlist ${nlist} \
--index_save_path ./experiments/index/hard_neg_prepare/ivf_nlist_${nlist}.index \