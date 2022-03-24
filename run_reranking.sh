data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/

export CUDA_VISIBLE_DEVICES="1"

python run_reranking.py \
--qid_2_query_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data/qid_2_query_token_ids.pkl \
--pid_2_passage_token_ids_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full/pid_2_passage_token_ids.pkl \
--re_rank_input_file_path /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/assets/msmarco/query_2_top_1000_passage_BM25.json \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_checkpoint /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/lightning_logs/single_GPU_full_1M_margin20/checkpoints/epoch=2-step=46874.ckpt \
--model_name bert_encoder \
--loss_name triplet_loss \
--output_file ranking.tsv > run_reranking.log 2>&1 &
echo $! > save_pid.txt