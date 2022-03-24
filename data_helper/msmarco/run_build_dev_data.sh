python build_passage_ranking_dev_data.py \
--tokenizer_name distilbert-base-uncased \
--query_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/queries.dev.small.tsv \
--truncate 128 \
--output_dir /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/dev_data > build_dev_data.log 2>&1 &
echo $! > save_pid.txt