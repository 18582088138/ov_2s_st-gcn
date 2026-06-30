import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()

        # Get device and dtype from input tensor (device-agnostic)
        device = x.device
        dtype = x.dtype

        # Create zero tensor on the same device as input
        zeros_start = torch.zeros(N, C, 1, V, M, device=device, dtype=dtype)

        # Compute motion difference
        motion_diff = x[:, :, 1:] - x[:, :, :-1]

        # Concatenate
        m = torch.cat((zeros_start, motion_diff), dim=2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res