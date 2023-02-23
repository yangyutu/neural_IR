data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/
pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"

python tasks/retrieval/ann_evaluation/run_retrieval_from_index.py \
--query_embedding_path ./experiments/precomputed_embeddings/query_embed.pkl \
--index_save_path ./experiments/index/exact_embed.index \
--output_path ./experiments/retrieval_results/out.json