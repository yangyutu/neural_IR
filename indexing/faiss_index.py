import enum
import sys
import time
from timeit import default_timer as timer

import faiss
import numpy as np
from tqdm import tqdm


def build_ann_faiss_index(
    data,
    dim,
    num_partitions,
    subquantizer_number,
    subquantizer_codebook_size,
    index_save_path,
    sub_sample_ratio=0.3,
    nprobe=10,
):

    start = time.time()
    sub_data = data[
        np.random.choice(
            data.shape[0], int(len(data) * sub_sample_ratio), replace=False
        )
    ]

    num_partitions = num_partitions  # size of the coarse codebook
    subquantizer_number = subquantizer_number
    subquantizer_codebook_size = subquantizer_codebook_size
    quantizer = faiss.IndexFlatL2(dim)
    # Inverted file with Product Quantizer encoding. Each residual vector is encoded as a product quantizer code.
    index = faiss.IndexIVFPQ(
        quantizer, dim, num_partitions, subquantizer_number, subquantizer_codebook_size
    )

    index.train(sub_data)
    index.add(data)
    index.nprobe = nprobe
    faiss.write_index(index, index_save_path)

    print(f"ANN faiss indexing finishing in {time.time() - start}")


def search_faiss_index(query_embed, index, number_nearest_nb):
    dis, ind = index.search(query_embed, number_nearest_nb)
    # print(dis)
    # print(ind)
    return ind, dis


def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[
                query_offset_base : query_offset_base + batch
            ]
            # index.search(batch_query_embeddings, topk) returns a tuple of two results
            # first result is the scores (from the nearest to the farest)
            # second result is the rank result
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(
        f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms"
    )
    return nearest_neighbors


def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    print("embedding shape: " + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        ids = ids.astype(np.int64)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index


gpu_resources = []


def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch

            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256 * 1024 * 1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index


if __name__ == "__main__":

    id_2_doc_embed_filename = (
        "/mnt/d/MLData/Repos/neural_IR/precomputed_embeddings/doc_embed.pkl"
    )
    id_2_query_embed_filename = (
        "/mnt/d/MLData/Repos/neural_IR/precomputed_embeddings/query_embed.pkl"
    )
    output_rank_file_path = "./retrieval_rank.tsv"
    import pickle, time

    gpu_flag = False
    with open(id_2_doc_embed_filename, "rb") as file:
        id_2_doc_embeds = pickle.load(file)

    with open(id_2_query_embed_filename, "rb") as file:
        id_2_query_embeds = pickle.load(file)

    doc_ids, doc_embeds = zip(*id_2_doc_embeds.items())
    query_ids, query_embeds = zip(*id_2_query_embeds.items())

    start = time.time()
    index = construct_flatindex_from_embeddings(np.array(doc_embeds), np.array(doc_ids))
    if gpu_flag:
        gpu_index = convert_index_to_gpu(
            index=index, faiss_gpu_index=0, useFloat16=True
        )
    print(f"index construction time: {time.time() - start}")

    nearest_nbs = index_retrieve(index, np.array(query_embeds), topk=1000, batch=128)

    with open(output_rank_file_path, "w") as outputfile:
        for qid, nb_list in zip(query_ids, nearest_nbs):
            for idx, nb_id in enumerate(nb_list):
                outputfile.write(f"{qid}\t{nb_id}\t{idx + 1}\n")
