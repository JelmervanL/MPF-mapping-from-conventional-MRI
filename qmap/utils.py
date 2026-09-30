import os
import random

import nibabel as nib
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode():
    """Use deterministic cuDNN/cuBLAS kernels (slower; makes GPU training runs repeatable)."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def select_device(gpu):
    """CUDA device `gpu`, or the CPU when gpu < 0."""
    if gpu is not None and gpu >= 0:
        torch.cuda.set_device(gpu)
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def save_nifti(array, affine, path):
    nib.save(nib.Nifti1Image(np.asarray(array).squeeze().astype(np.float32), affine), path)
