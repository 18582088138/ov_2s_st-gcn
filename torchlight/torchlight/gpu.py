import os
import torch


def visible_gpu(gpus):
    """
        set visible gpu.

        can be a single id, or a list

        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, gpus)))
    return list(range(len(gpus)))


def ngpu(gpus):
    """
        count how many gpus used.
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


def occupy_gpu(gpus=None):
    """
        make program appear on nvidia-smi.

        Modified to check CUDA availability before occupying GPU.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return  # Silently return if no CUDA available

    if gpus is None:
        torch.zeros(1).cuda()
    else:
        gpus = [gpus] if isinstance(gpus, int) else list(gpus)
        for g in gpus:
            # Check if GPU index is valid
            if g < torch.cuda.device_count():
                torch.zeros(1).cuda(g)
