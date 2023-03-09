retrieval_result_path=./experiments/retrieval_results/simple_neg/out.json

python tasks/retrieval/ann_evaluation/run_eval_retrieval_results.py \
--ground_truth_path ./assets/msmarco/query_2_groundtruth_passage_small.json \
--candidate_path ${retrieval_result_path=./experiments/retrieval_results/simple_neg/out.json}