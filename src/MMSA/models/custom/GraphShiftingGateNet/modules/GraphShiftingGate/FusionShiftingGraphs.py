import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict   

from ..GraphAttentions import GraphAttention, SinusoidalPositionalEmbedding

__all__ = ['FusionShiftingGraph']


class FusionShiftingGraph(nn.Module):
    def __init__(self, cfg):
        super(FusionShiftingGraph, self).__init__()
        self.cfg = cfg
        self.inter_graph = InterGraph(cfg)
        self.intra_graph = IntraGraph(cfg)
    
    def forward(self, inter_cat_seq, inter_split):
        inter_output = self.inter_graph(inter_cat_seq, inter_split)
        intra_output = self.intra_graph(inter_cat_seq, inter_output)
        return inter_output, intra_output
    

class InterGraph(nn.Module):
    def __init__(self, cfg):
        super(InterGraph, self).__init__()
        self.cfg = cfg
        self.embed_scale = math.sqrt(cfg.text_embed)
        self.pos_embeds = nn.ModuleList([
            SinusoidalPositionalEmbedding(cfg.text_embed)
            for _ in cfg.inter_seq_lens
        ])
        self.intra_mask = self.get_intra_mask_0_1(cfg.inter_seq_lens)
        self.inter_modal_graphs = nn.ModuleList([GraphAttention(
            cfg.text_embed, cfg.inter_num_heads, cfg.inter_random_masking_rate
        ) for _ in range(cfg.num_inter_graph_layers)])
        self.layer_norm = nn.LayerNorm(cfg.text_embed)

    def forward(self, inter_cat_seq, inter_split):
        pos_encoded_nodes = torch.concat([
            self.get_pos_encoded_seq(pos_embed_i, seq_i)
            for pos_embed_i, seq_i in zip(self.pos_embeds, inter_cat_seq)
        ], dim=1)
        modal_intra_mask = self.get_intra_mask_neginf_0().to(pos_encoded_nodes.device) # Put the edge masking into GraphAttention Module
        
        temp_nodes = pos_encoded_nodes.permute(1, 0, 2)
        for graph in self.inter_modal_graphs:
            attned_nodes, _ = graph(
                query_nodes=temp_nodes, key_nodes=temp_nodes, value_nodes=temp_nodes, 
                edge_mask=modal_intra_mask
            )
            temp_nodes = attned_nodes
        output = self.layer_norm(attned_nodes.permute(1, 0, 2))
        return EasyDict({
            'output_tensor': output, 
            'split_output': list(torch.split(output, inter_split, dim=1))
        })

    def get_intra_mask_neginf_0(self, random_masking_rate=0):
        neg_inf = -10e9
        intra_mask = F.dropout(
            self.intra_mask, 
            p=random_masking_rate, 
            training=self.training
        )
        return torch.where(
            intra_mask == 0, neg_inf,
            torch.tensor(0, dtype=torch.float32)
        )

    def get_intra_mask_0_1(self, inter_split):
        temp = 0
        sum_len = sum(inter_split)
        mask_list = []
        for split_len in inter_split:
            for _ in range(split_len):
                row_mask_tensor = torch.ones(sum_len, dtype=torch.float32)
                row_mask_tensor[temp:temp + split_len] = 0
                mask_list.append(row_mask_tensor)
            temp += split_len
        return torch.stack(mask_list)

    def get_pos_encoded_seq(self, pos_embed, seq):
        return F.dropout(
            (self.embed_scale * seq + 
             pos_embed(seq.transpose(0, 1)[:, :, 0]).transpose(0, 1)),
            p=self.cfg.inter_pos_dropout, training=self.training
        )


class IntraGraph(nn.Module):
    def __init__(self, cfg):
        super(IntraGraph, self).__init__()
        self.cfg = cfg
        self.embed_scale = math.sqrt(cfg.v_feat_out)
        self.pos_embeds = nn.ModuleList([
            SinusoidalPositionalEmbedding(cfg.v_feat_out)
            for _ in cfg.inter_seq_lens
        ])
        self.intra_mask = self.get_inter_mask_0_1(cfg.intra_seq_lens)
        self.intra_modal_graphs = nn.ModuleList([GraphAttention(
            cfg.v_feat_out, cfg.intra_num_heads, cfg.intra_random_masking_rate
        ) for _ in range(cfg.num_intra_graph_layers)])
        self.layer_norm = nn.LayerNorm(cfg.v_feat_out)
    
    def forward(self, intra_cat_seq, intra_split):
        pos_encoded_nodes = torch.concat([
            self.get_pos_encoded_seq(pos_embed_i, seq_i)
            for pos_embed_i, seq_i in zip(self.pos_embeds, intra_cat_seq)
        ], dim=1)
        modal_intra_mask = self.get_inter_mask_neginf_0().to(pos_encoded_nodes.device) # Put the edge masking into GraphAttention Module
        
        temp_nodes = pos_encoded_nodes.permute(1, 0, 2)
        for graph in self.intra_modal_graphs:
            attned_nodes, _ = graph(
                query_nodes=temp_nodes, key_nodes=temp_nodes, value_nodes=temp_nodes, 
                edge_mask=modal_intra_mask
            )
            temp_nodes = attned_nodes
        output = self.layer_norm(attned_nodes.permute(1, 0, 2))
        return EasyDict({
            'output_tensor': output, 
            'split_output': list(torch.split(output, intra_split, dim=1))
        })

    def get_inter_mask_neginf_0(self, random_masking_rate=0):
        neg_inf = -10e9
        intra_mask = F.dropout(
            self.intra_mask, 
            p=random_masking_rate, 
            training=self.training
        )
        return torch.where(
            intra_mask == 0, neg_inf,
            torch.tensor(0, dtype=torch.float32)
        )
    
    def get_inter_mask_0_1(self, intra_split):
        temp = 0
        sum_len = sum(intra_split)
        mask_list = []
        for split_len in intra_split:
            for _ in range(split_len):
                row_mask_tensor = torch.ones(sum_len, dtype=torch.float32)
                row_mask_tensor[temp:temp + split_len] = 0
                mask_list.append(row_mask_tensor)
            temp += split_len
        return torch.abs(torch.stack(mask_list) - 1)
    
    def get_pos_encoded_seq(self, pos_embed, seq):
        return F.dropout(
            (self.embed_scale * seq + 
             pos_embed(seq.transpose(0, 1)[:, :, 0]).transpose(0, 1)),
            p=self.cfg.intra_pos_dropout, training=self.training
        )