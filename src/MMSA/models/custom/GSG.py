import torch
import torch.nn as nn

from .modules import GATv2Conv_PyG

class GSG_BERT(nn.Module):
    def __init__(self, args):
        super(GSG_BERT, self).__init__()
        