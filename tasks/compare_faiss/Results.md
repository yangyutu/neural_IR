
## exact cpu 24 core
41m17

mrr: {'MRR @10': 0.28814776913630685, 'QueriesRanked': 6980}
recall_5: 0.42438
recall_20: 0.62784
recall_100: 0.80618

## ivf cpu 24 core
nprobe 10
44 s

mrr: {'MRR @10': 0.25581161595488155, 'QueriesRanked': 6980}
recall_5: 0.37009
recall_20: 0.53789
recall_100: 0.67553

nprobe 30
1m32.631s

mrr: {'MRR @10': 0.2743857165597843, 'QueriesRanked': 6980}
recall_5: 0.4005
recall_20: 0.59064
recall_100: 0.75165

nprobe 100
4m23.264s

mrr: {'MRR @10': 0.2851743643971428, 'QueriesRanked': 6980}
recall_5: 0.41908
recall_20: 0.62001
recall_100: 0.79207

## ivfpq cpu 24 core subquantizer_number=8 subquantizer_codebook_size=8

nprobe 10
3.556s

mrr: {'MRR @10': 0.13216912266339245, 'QueriesRanked': 6980}
recall_5: 0.19784
recall_20: 0.33179
recall_100: 0.48541

nprobe 30
3.927s

mrr: {'MRR @10': 0.1334497089189071, 'QueriesRanked': 6980}
recall_5: 0.19999
recall_20: 0.3376
recall_100: 0.49936

nprobe 100
6.600s
mrr: {'MRR @10': 0.13354698230772774, 'QueriesRanked': 6980}
recall_5: 0.19999
recall_20: 0.33817
recall_100: 0.50165

nprobe 500

20.885s
mrr: {'MRR @10': 0.1335398189839455, 'QueriesRanked': 6980}
recall_5: 0.19999
recall_20: 0.33817
recall_100: 0.50179

## ivfpq cpu 24 core subquantizer_number=16 subquantizer_codebook_size=8

nprobe 10
3.56s

mrr: {'MRR @10': 0.18467276117705908, 'QueriesRanked': 6980}
recall_5: 0.27681
recall_20: 0.43164
recall_100: 0.58669

nprobe 30
4.61s

mrr: {'MRR @10': 0.19064316414244756, 'QueriesRanked': 6980}
recall_5: 0.2872
recall_20: 0.45451
recall_100: 0.62699

nprobe 100
9.03s

mrr: {'MRR @10': 0.1918421680993311, 'QueriesRanked': 6980}
recall_5: 0.28964
recall_20: 0.45996
recall_100: 0.63931

### storage savings

exact index: 12.7 GB
ivfpq_8_8: 136 MB
ivfpq_8_8: 204 MB