pretrained_model_name="distilbert-base-uncased"
pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
model_checkpoint="artifacts/3m2j7sqb/model_v0.ckpt"
#bert-base-uncased
#sentence-transformers/msmarco-distilbert-base-v4
export CUDA_VISIBLE_DEVICES="0"
python tasks/retrieval/inference/run_precompute_embeddings.py \
--input_filename /mnt/d/MLData/data/msmarco_passage/queries.dev.small.tsv \
--output_filename ./experiments/precomputed_embeddings/query_embed.pkl \
--pretrained_model_name ${pretrained_model_name} \
--model_checkpoint ${model_checkpoint} \
--query \
--truncate 128

# may take ~1hr for small model like microsoft/MiniLM-L12-H384-uncased
python tasks/retrieval/inference/run_precompute_embeddings.py \
--input_filename /mnt/d/MLData/data/msmarco_passage/collection_dev_only.tsv \
--output_filename ./experiments/precomputed_embeddings/doc_embed.pkl \
--pretrained_model_name ${pretrained_model_name} \
--model_checkpoint ${model_checkpoint} \
--truncate 128
