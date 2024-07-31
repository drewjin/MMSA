import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ShiftingGate']

class ShiftingGate(nn.Module):
    def __init__(
        self, 
        cfg, 
        embed_dim_1,
        embed_dim_2, 
        beta_shift=1, 
        shifting_gate_dropout_prob=0.1
    ):
        super(ShiftingGate, self).__init__()
        self.cfg = cfg

        self.W_h = nn.Linear(embed_dim_1 + embed_dim_2, embed_dim_1)

        self.W_m = nn.Linear(embed_dim_2, embed_dim_1)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(embed_dim_1)
        self.dropout = nn.Dropout(shifting_gate_dropout_prob)

    def forward(self, text_embed, text_fused):
        eps = 1e-6
        weight_h = F.relu(self.W_h(torch.cat((text_embed, text_fused), dim=-1)))
        h_m = weight_h * self.W_m(text_fused)

        em_norm = text_embed.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embed.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embed.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        fused_text_embed = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(fused_text_embed + text_embed)
        )

        return embedding_output
    
    