data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/

export CUDA_VISIBLE_DEVICES="5"

python run_retrieval.py \
--qid_2_query_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data/qid_2_query_token_ids.pkl \
--pid_2_embedding_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/passage_embedding.pkl \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/version_12/checkpoints/epoch=1-step=312499.ckpt \
--index_save_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/embed.index \
--number_nearest_neighbors 5 \
--output_file retrieval.tsv \
--model_name bert_encoder \
--loss_name triplet_loss > run_retreival.log 2>&1 &
echo $! > save_pid.txt