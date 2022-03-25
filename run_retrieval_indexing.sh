data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/

export CUDA_VISIBLE_DEVICES="6"

python run_retrieval_indexing.py \
--pid_2_passage_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full/pid_2_passage_token_ids.pkl \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--embedding_save_path passage_embedding.pkl \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/version_14/checkpoints/epoch=0-step=156249.ckpt \
--num_partitions 2000 \
--subquantizer_number 8 \
--subquantizer_codebook_size 8 \
--model_name bert_encoder \
--loss_name triplet_loss \
--index_save_path embed.index > run_retreival_indexing.log 2>&1 &
echo $! > save_pid.txt