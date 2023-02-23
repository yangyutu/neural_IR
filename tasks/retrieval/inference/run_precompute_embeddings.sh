pretrained_model_name="distilbert-base-uncased"
model_checkpoint="artifacts/bi_encoder_model_distilbert-33pjv1vn:v1/model.ckpt"
#bert-base-uncased
#sentence-transformers/msmarco-distilbert-base-v4
export CUDA_VISIBLE_DEVICES="0"
python run_precompute_embeddings.py \
--input_filename /mnt/d/MLData/data/msmarco_passage/queries.dev.small.tsv \
--output_filename ./precomputed_embeddings/query_embed.pkl \
--pretrained_model_name ${pretrained_model_name} \
--model_checkpoint ${model_checkpoint} \
--truncate 128

python run_precompute_embeddings.py \
--input_filename /mnt/d/MLData/data/msmarco_passage/collection_dev_only.tsv \
--output_filename ./precomputed_embeddings/doc_embed.pkl \
--model_checkpoint ${model_checkpoint} \
--truncate 128
