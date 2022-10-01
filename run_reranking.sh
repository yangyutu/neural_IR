data_root=./experiments/msmarco_psg_ranking/triplet_train_data_tiny/
dev_query_root=./experiments/msmarco_psg_ranking/dev_data/
output_dir=./experiments/msmarco_psg_ranking/evaluation/
model_checkpoint=/mnt/d/MLData/Repos/neural_IR/experiments/msmarco_psg_ranking/logs/lightning_logs/version_0/checkpoints/epoch=3-step=25128.ckpt
export CUDA_VISIBLE_DEVICES="0"

python run_reranking.py \
--qid_2_query_token_ids_path ${dev_query_root}qid_2_query_token_ids.pkl \
--pid_2_passage_token_ids_path ${data_root}pid_2_passage_token_ids.pkl \
--re_rank_input_file_path ./assets/msmarco/query_2_top_1000_passage_BM25.json \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_checkpoint ${model_checkpoint} \
--model_name bert_encoder \
--loss_name triplet_loss \
--output_file ${output_dir}ranking.tsv 2>&1 | tee run_reranking.log
#echo $! > save_pid.txt