import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict

from .ShiftingGates import ShiftingGate
from .FusionShiftingGraphs import FusionShiftingGraph
from ..TempModels import xLSTMProj
from ..utils import XLSTMCFG


__all__ = ['GSG']


class GSG(nn.Module):
    def __init__(self, args):
        super(GSG, self).__init__()
        self.args = args
        self.config = cfg = args.config
        # Project the tensors into the same dimension, 
        # then concatenate them in the temporal dimension
        
        # Inter Projection
        self.v_res_proj_1  = nn.Linear(cfg.v_res_in, cfg.text_embed)
        self.a_d2v_proj_1  = nn.Linear(cfg.a_d2v_in, cfg.text_embed)
        # self.v_feat_proj_1 = nn.Linear(cfg.v_feat_in, cfg.text_embed)
        # self.a_feat_proj_1 = nn.Linear(cfg.a_feat_in, cfg.text_embed)

        # Intra Projection
        # visual, audio features are con..xLSTM import xLSTMBlockStack, xLSTMBlockStackConfigsidered as the same modality separately
        # self.v_res_proj_2  = nn.Linear(cfg.v_res_in, cfg.v_feat_out)
        # self.a_d2v_proj_2  = nn.Linear(cfg.a_d2v_in, cfg.a_feat_out)
        # self.v_feat_proj_2 = nn.Linear(cfg.v_feat_in, cfg.v_feat_out)
        # self.a_feat_proj_2 = nn.Linear(cfg.a_feat_in, cfg.a_feat_out)

        self.shifting_graph = FusionShiftingGraph(cfg)

        self.text_shifting_gate = ShiftingGate(
            cfg, cfg.text_embed, cfg.text_embed, 
            cfg.beta_shift, cfg.shifting_gate_dropout_prob
        )
        
        # inter_modules = len(cfg.inter_seq_lens) - 1 # except text
        # self.inter_xlstms = self.get_temporal_models(
        #     [cfg.text_embed for _ in range(inter_modules)],
        #     [cfg.inter_out for _ in range(inter_modules)],
        #     cfg.inter_seq_lens[1:]
        # )

        # intra_modules = len(cfg.intra_seq_lens)
        # self.intra_xlstms = self.get_temporal_models(
        #     [cfg.a_feat_out for _ in range(intra_modules)],
        #     [cfg.intra_out for _ in range(intra_modules)],
        #     cfg.intra_seq_lens
        # )

        # self.v_shifting_gate = ShiftingGate(
        #     cfg, 2*cfg.intra_out, 2*cfg.inter_out, 
        #     cfg.beta_shift, cfg.shifting_gate_dropout_prob
        # )
        # self.a_shifting_gate = ShiftingGate(
        #     cfg, 2*cfg.intra_out, 2*cfg.inter_out, 
        #     cfg.beta_shift, cfg.shifting_gate_dropout_prob
        # )

    def forward(self, text_embed, visual_res, audio_d2v):
        # Inter Projection
        v_res_embed = self.v_res_proj_1(visual_res)
        a_d2v_embed = self.a_d2v_proj_1(audio_d2v)
        # v_feat_embed = self.v_feat_proj_1(visual_feat)
        # a_feat_embed = self.a_feat_proj_1(audio_feat)
        inter_cat_seq = [text_embed, v_res_embed, a_d2v_embed]
        inter_split = self.get_seq_split(inter_cat_seq)
        
        # # Intra Projection
        # v_res_1 = self.v_res_proj_2(visual_res)
        # v_feat_1 = self.v_feat_proj_2(visual_feat)
        # v_cat_seq = [v_res_1, v_feat_1]

        # a_d2v_1 = self.a_d2v_proj_2(audio_d2v)
        # a_feat_1 = self.a_feat_proj_2(audio_feat)
        # a_cat_seq = [a_d2v_1, a_feat_1]

        # intra_cat_seq = v_cat_seq + a_cat_seq
        # intra_split = self.get_seq_split(intra_cat_seq)

        # Shifting
        inter_fused, intra_fused = self.shifting_graph(
            inter_cat_seq, inter_split, 
            # intra_cat_seq, intra_split
        )
        text_fused = inter_fused.split_output[0]
        text_shifted = self.text_shifting_gate(text_embed, text_fused)

        # # Pool the Fused Output of Other Modals Using xLSTM
        # inter_pooled = [] # [v_res, a_d2v, v_feat, a_feat]
        # for idx, (fused_modal_i, xlstm) in enumerate(zip(inter_fused.split_output[1:], self.inter_xlstms)):
        #     pooled_modal_i = xlstm(fused_modal_i)
        #     inter_pooled.append(pooled_modal_i)
        
        # intra_pooled = [] # [v_res, v_feat, a_d2v, a_feat]
        # for idx, (fused_modal_i, xlstm) in enumerate(zip(intra_fused.split_output, self.intra_xlstms)):
        #     pooled_modal_i = xlstm(fused_modal_i)
        #     intra_pooled.append(pooled_modal_i)

        # # Inter_Fused Pooled Output
        # inter_v = torch.cat([inter_pooled[0], inter_pooled[2]], dim=-1)
        # inter_a = torch.cat([inter_pooled[1], inter_pooled[3]], dim=-1)
        # # Intra_Enhanced Pooled Output
        # intra_v = torch.cat([intra_pooled[0], intra_pooled[1]], dim=-1)
        # intra_a = torch.cat([intra_pooled[2], intra_pooled[3]], dim=-1)

        # visual_shifted = self.v_shifting_gate(intra_v, inter_v)
        # audio_shifted = self.a_shifting_gate(intra_a, inter_a)

        return EasyDict({
            "text_shifted": text_shifted,
            # "visual_shifted": visual_shifted,
            # "audio_shifted": audio_shifted,
            # "inter_pooled": inter_pooled,
            # "intra_pooled": intra_pooled,
            "inter_fused": inter_fused,
            "intra_fused": intra_fused
        })

    
    def get_seq_split(self, seq_list):
        return [seq.shape[1] for seq in seq_list]
    

    def get_temporal_models(
        self, 
        in_embeds, 
        out_embeds, 
        split_list
    ):
        return nn.ModuleList([
            xLSTMProj(XLSTMCFG(
                in_embed=in_embed, 
                out_embed=out_embed, 
                context_length=split,
                num_blocks=2
            ))
            for in_embed, out_embed, split in zip(
                in_embeds, out_embeds, split_list
            )
        ])
        