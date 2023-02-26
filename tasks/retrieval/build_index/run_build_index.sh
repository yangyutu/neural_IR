data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/
pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"

python tasks/retrieval/build_index/run_build_index.py \
--doc_embedding_path ./experiments/precomputed_embeddings/doc_embed.pkl \
--index_save_path ./experiments/index/exact_embed.index \
--index_type exact


