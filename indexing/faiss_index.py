import faiss
import numpy as np
import time


def build_faiss_index(
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

    print(f"faiss indexing finishing in {time.time() - start}")


def search_faiss_index(query_embed, index, number_nearest_nb):
    dis, ind = index.search(query_embed, number_nearest_nb)
    # print(dis)
    # print(ind)
    return ind, dis

