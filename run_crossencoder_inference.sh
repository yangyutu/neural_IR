val_data_root=./experiments/msmarco_psg_ranking/dev_data_sz_full/
export CUDA_VISIBLE_DEVICES="0"
python run_crossencoder_inference.py \
--gpus 1 \
--batch_size 256 \
--num_workers 16 \
--pretrained_model_name bert-base-uncased \
--model_checkpoint artifacts/model-2zhakltb:v3/model.ckpt \
--pid_2_passage_path ${val_data_root}pid_2_passage_text.pkl \
--qid_2_query_path ${val_data_root}qid_2_query_text.pkl \
--query_candidates_path ${val_data_root}qid_2_top_1000_passage_BM25_subset.json \
--qrels_path ./assets/msmarco/query_2_groundtruth_passage_small.json \
--max_len 128 \
--project_name crossencoder_rerank_MSMARCO_bert_evaluation \
--default_root_dir ./experiments/msmarco_psg_ranking/logs 2>&1 | tee run_cross_encoder_rerank_inference.log
echo $! > save_pid.txt