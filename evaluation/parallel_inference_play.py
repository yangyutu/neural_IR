import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from utils.distributed_sampler import DistributedEvalSampler
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# from multiprocessing import Process, Queue

# def run_inference(rank, world_size):
#     # create default process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
#     # load a model 
#     model = YourModel()
#     model.load_state_dict(PATH)
#     model.eval()
#     model.to(rank)

#     # create a dataloader
#     dataset = ...
#     loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                num_workers=4)

#     # iterate over the loaded partition and run the model
#     for idx, data in enumerate(loader):
#             ...

class Indentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def run_inference(rank, world_size, model, dataset, return_dict):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    

 
    # create a dataloader

    sampler = DistributedSampler(dataset)
    sampler = DistributedEvalSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=4,
                                               num_workers=4, 
                                               sampler=sampler)

    output = {}
    # iterate over the loaded partition and run the model
    for idx, data in enumerate(loader):
        out = model(data)
        print(f'batch idx {idx}, output {out} at rank {rank}')
        output[(idx, rank)] = out

    return_dict.update(output)

def run_inference_toy(rank, world_size, return_dict):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # load a model 
    model = Indentity()
    model.eval()
 
    # create a dataloader
    inps = torch.arange(10 * 4, dtype=torch.float32)
    dataset = TensorDataset(inps)
    sampler = DistributedSampler(dataset)
    sampler = DistributedEvalSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=4,
                                               num_workers=4, 
                                               sampler=sampler)

    output = {}
    # iterate over the loaded partition and run the model
    for idx, data in enumerate(loader):
        out = model(data)
        print(f'batch idx {idx}, output {out} at rank {rank}')
        output[(idx, rank)] = out

    return_dict.update(output)

def run_hello(rank, world_size):


    print(f'hello from {rank} !')


def main():
    world_size = 4

    inps = torch.arange(10 * 4, dtype=torch.float32)
    dataset = TensorDataset(inps)

    # load a model 
    model = Indentity()
    model.eval()

    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(run_inference,
        args=(world_size, model, dataset, return_dict, ),
        nprocs=world_size,
        join=True)


    # manager = mp.Manager()
    # return_dict = manager.dict()
    # mp.spawn(run_inference_toy,
    #     args=(world_size, return_dict),
    #     nprocs=world_size,
    #     join=True)

    for k, v in return_dict.items():
        print(k, v)

    # mp.spawn(run_hello,
    #     args=(world_size, ),
    #     nprocs=world_size,
    #     join=True)

if __name__=="__main__":
    main()