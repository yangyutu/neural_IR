train_data_root=./experiments/msmarco_psg_ranking/cross_encoder_triplet_train_data_medium_mixed/
val_data_root=./experiments/msmarco_psg_ranking/dev_data_sz_1000/
pretrained_model_name="bert-base-uncased"
pretrained_model_name="sentence-transformers/msmarco-distilbert-base-v4"
pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="nreimers/MiniLM-L6-H384-uncased"
pretrained_model_name="nreimers/MiniLM-L3-H384-uncased"
teacher_model_ckpt=artifacts/3m2j7sqb/model_v0.ckpt
distill_method=listnet_loss
#bert-base-uncased
#sentence-transformers/msmarco-distilbert-base-v4
export CUDA_VISIBLE_DEVICES="0"
python tasks/retrieval_knowledge_distillation/training/run_biencoder_training_with_distill.py \
--gpus 1 \
--limit_train_batches 1.0 \
--max_epochs 3 \
--model_save_every_n_steps 40000 \
--model_validate_every_n_steps 40000 \
--lr 3e-6 \
--batch_size 64 \
--infer_batch_size 256 \
--num_workers 24 \
--pretrained_model_name ${pretrained_model_name} \
--teacher_model_ckpt ${teacher_model_ckpt} \
--distill_loss_type ${distill_method} \
--distill_loss_coeff 1.0 \
--contrast_loss_coeff 1.0 \
--train_triplet_path ${train_data_root}triplets.pkl \
--train_pid_2_passage_path ${train_data_root}pid_2_passage_text.pkl \
--train_qid_2_query_path ${train_data_root}qid_2_query_text.pkl \
--val_pid_2_passage_path ${val_data_root}pid_2_passage_text.pkl \
--val_qid_2_query_path ${val_data_root}qid_2_query_text.pkl \
--val_query_candidates_path ${val_data_root}qid_2_top_1000_passage_BM25_subset.json \
--val_qrels_path ./assets/msmarco/query_2_groundtruth_passage_small.json \
--max_len 128 \
--project_name biencoder_retrieval_MSMARCO_bert \
--tag knowledge_distill,${distill_method} \
--default_root_dir ./experiments/msmarco_psg_ranking/logs 2>&1 | tee run_bi_encoder_rerank_training.log
echo $! > save_pid.txt