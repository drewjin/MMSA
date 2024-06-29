import torch
import torch.nn as nn

class Decomposer(nn.Module):
    """
        Decomposer: Decompose the input into activities
    """
    pass

class DecomposeAttention(nn.Module):
    """
        DecomposeAttention: Decomposed Features were used as input to the attention mechanism
        - Ordinary dot product attention (v1)
        - Multi-head attention (v2)
    """
    pass

class EnhanceNet_v1(nn.Module):
    """
        EnhanceNet_v1: Using only Decomposer and DecomposeAttention(none Multi-Head version)
    """
    pass

class EnhanceNet_v2(nn.Module):
    """
        EnhanceNet_v2:  
        - Decomposer
        - DecomposeAttention(Multi-Head version)
        - GlobalMultiHeadAttention and ResidualConnection
    """
    pass
