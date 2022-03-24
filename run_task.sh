data_root=/home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_small/

python run_task.py \
--gpus 1 \
--limit_train_batches 1.0 \
--max_epochs 30 \
--lr 2e-6 \
--batch_size 64 \
--margin 20.0 \
--tokenizer_name distilbert-base-uncased \
--pretrained_model_name distilbert-base-uncased \
--model_name bert_encoder \
--loss_name triplet_loss \
--triplet_path ${data_root}triplets.pkl \
--pid_2_passage_token_ids_path ${data_root}pid_2_passage_token_ids.pkl \
--qid_2_query_token_ids_path ${data_root}qid_2_query_token_ids.pkl \
--max_len 128 \
--default_root_dir /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments > run_task.log 2>&1 &
echo $! > save_pid.txt