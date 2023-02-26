doc_embed_path=./experiments/precomputed_embeddings/hard_neg_prepare/doc_embed.pkl

# exact
# python tasks/compare_faiss/run_build_different_indexing.py \
# --doc_embedding_path ${doc_embed_path} \
# --index_type exact \
# --index_save_path ./experiments/index/faiss_compare/exact_embed.index \

# ivf
# nlist=1000
# python tasks/compare_faiss/run_build_different_indexing.py \
# --doc_embedding_path ${doc_embed_path} \
# --index_type ivf \
# --data_frac_for_training 0.1 \
# --nlist ${nlist} \
# --index_save_path ./experiments/index/faiss_compare/ivf_nlist_${nlist}.index \

# ivfpq
nlist=1000
subquantizer_number=16
subquantizer_codebook_size=8
python tasks/compare_faiss/run_build_different_indexing.py \
--doc_embedding_path ${doc_embed_path} \
--index_type ivfpq \
--data_frac_for_training 0.1 \
--nlist ${nlist} \
--subquantizer_number ${subquantizer_number} \
--subquantizer_codebook_size ${subquantizer_codebook_size} \
--index_save_path ./experiments/index/faiss_compare/ivfpq_nlist_${nlist}_subnum_${subquantizer_number}_subcbsize_${subquantizer_codebook_size}.index \
