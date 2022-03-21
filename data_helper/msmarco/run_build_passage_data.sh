python build_passage_triplet_train_data.py \
--tokenizer_name distilbert-base-uncased \
--triplet_file /home/ubuntu/MLData/work/Repos/NeuralIR/data/qidpidtriples.train.full.2.tsv \
--passage_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/collection.tsv \
--query_collection /home/ubuntu/MLData/work/Repos/NeuralIR/data/collectionandqueries/queries.train.tsv \
--truncate 128 \
--output_dir /home/ubuntu/MLData/work/Repos/NeuralIR/neural_IR/experiments/msmarco_psg/train_data_full_1000000 > build_train_data.log 2>&1 &
echo $! > save_pid.txt