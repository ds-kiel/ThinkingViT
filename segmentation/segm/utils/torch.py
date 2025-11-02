import os
import torch


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)),
        )
    )
    dist_rank = int(
        os.environ.get(
            "RANK",
            os.environ.get("SLURM_PROCID", os.environ.get("OMPI_COMM_WORLD_RANK", 0)),
        )
    )
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)),
        )
    )

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
