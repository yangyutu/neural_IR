import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.distributed_sampler import DistributedEvalSampler
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def parallel_encoding(rank, world_size, model, dataset, return_dict, batch_size=32, num_workers=4):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # distributed sampler is use to partition the dataset into subgroups and each subgroup will run on one process
    sampler = DistributedEvalSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers, 
                                               sampler=sampler)

    output = {}
    print(f' starting encoding from {rank}')
    # iterate over the loaded partition and run the model
    for idx, batch in enumerate(data_loader):
        print(f'batch {idx} from {rank}')
        data_label, data = batch
        model_out = model(data)
        output[(idx, rank)] = (data_label, model_out)

    print(f' finishing encoding from {rank}')
    return_dict.update(output)    

