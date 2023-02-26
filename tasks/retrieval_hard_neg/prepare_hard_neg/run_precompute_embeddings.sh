
model_checkpoint="artifacts/3m2j7sqb/model_v0.ckpt"
export CUDA_VISIBLE_DEVICES="0"
python tasks/retrieval/inference/run_precompute_embeddings.py \
--input_filename /mnt/d/MLData/data/msmarco_passage/queries.train.tsv \
--output_filename ./experiments/precomputed_embeddings/hard_neg_prepare/query_embed.pkl \
--model_checkpoint ${model_checkpoint} \
--query \
--truncate 128

# may take ~1hr for small model like microsoft/MiniLM-L12-H384-uncased
# python tasks/retrieval/inference/run_precompute_embeddings.py \
# --input_filename /mnt/d/MLData/data/msmarco_passage/collection.tsv \
# --output_filename ./experiments/precomputed_embeddings/hard_neg/doc_embed.pkl \
# --model_checkpoint ${model_checkpoint} \
# --truncate 128
