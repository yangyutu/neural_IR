data_root=/mnt/d/MLData/data/msmarco_passage/
python data_helper/msmarco/build_passage_ranking_dev_data.py \
--passage_collection ${data_root}collection.tsv \
--query_collection ${data_root}queries.dev.small.tsv \
--query_candidates_path ./assets/msmarco/query_2_top_1000_passage_BM25.json \
--sample_size 500 \
--output_dir ./experiments/msmarco_psg_ranking/dev_data_sz_500 2>&1 | tee dev_data.log
#echo $! > save_pid.txt