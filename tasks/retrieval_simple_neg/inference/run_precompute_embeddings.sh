
model_checkpoint="artifacts/3m2j7sqb/model_v0.ckpt"
query_file=/mnt/d/MLData/data/msmarco_passage/queries.dev.small.tsv
doc_file=/mnt/d/MLData/data/msmarco_passage/collection.tsv
query_embedding_output_path=./experiments/precomputed_embeddings/simple_neg/query_embed.pkl
doc_embedding_output_path=./experiments/precomputed_embeddings/simple_neg/doc_embed.pkl

#bert-base-uncased
#sentence-transformers/msmarco-distilbert-base-v4
export CUDA_VISIBLE_DEVICES="0"
python tasks/retrieval/inference/run_precompute_embeddings.py \
--input_filename ${query_file} \
--output_filename ${query_embedding_output_path} \
--model_checkpoint ${model_checkpoint} \
--query \
--truncate 128

# may take ~1hr for small model like microsoft/MiniLM-L12-H384-uncased
python tasks/retrieval/inference/run_precompute_embeddings.py \
--input_filename ${doc_file} \
--output_filename ${doc_embedding_output_path} \
--model_checkpoint ${model_checkpoint} \
--truncate 128
