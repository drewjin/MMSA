import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from transformers import Data2VecAudioModel, BertTokenizer, BertModel

from .modules import TempResNet, xLSTMProj, NaiveLSTMProj, GSG
from .modules.utils import XLSTMCFG


__all__ = ['GSGNET']
    

class GSGNET(nn.Module):
    def __init__(self, args):
        super(GSGNET, self).__init__()
        cfg = args.config
        # TEXT TOKENIZITION
        self.aligned = args.need_data_aligned

        # AUDIO EXTRACTION
        # Using Data2Vec to extract audio temporal features
        self.audio_model = Data2VecAudioModel.from_pretrained(args.data2vec)
        # Using LSTMs to extract audio features
        self.use_xlstm = args.use_xlstm
        if args.use_xlstm:
            self.audio_d2v_xlstm_proj = xLSTMProj(
                XLSTMCFG(
                    context_length=cfg.inter_seq_lens[2],
                    in_embed=args.audio_embed, 
                    out_embed=args.audio_embed, 
                    num_blocks=args.a_lstm_layers, 
                    lstm_dropout=args.a_lstm_dropout
                ))
        else:
            # TODO
            self.audio_proj = NaiveLSTMProj(
                args.audio_embed, args.a_lstm_hidden_size, args.audio_out, 
                num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout
            )

        # VISION EXTRACTION
        # Using TempResNet to extract visual features
        self.video_model = TempResNet(args)

        
        # GRAPH SHIFTING GATE NETWORK
        self.graph_shifting_gate = GSG(args)


        # BERT MODEL
        self.bert_model = BertModel.from_pretrained(args.weight_dir)
        # self.bert_embedding = self.bert_model.get_input_embeddings()


        # SELF SUPERVISION ON ENCODED FEATURES
        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(cfg.text_embed + 4*cfg.intra_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(cfg.text_embed, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim//2)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim//2, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(2*cfg.intra_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim//2)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim//2, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(2*cfg.intra_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim//2)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim//2, 1)


        # Classifier
        # self.text_classifier = nn.Sequential(
        #     nn.Dropout(args.cls_dropout),
        #     nn.Linear(cfg.text_embed, cfg.text_embed),
        #     nn.ReLU(),
        #     nn.Linear(cfg.text_embed, cfg.text_embed//2),
        #     nn.ReLU(),
        #     nn.Linear(cfg.text_embed//2, 1)
        # )


    def forward(self, text, audio, video):
        audio_wav, audio_lengths = audio
        video_rgb, video_lengths = video

        # mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        # text_lengths = mask_len.squeeze(1).int().detach().cpu()
        input_ids, input_mask, segment_ids = text
        # text_embedding = self.bert_embedding(input_ids)
        text = self.bert_model(input_ids=input_ids)
        text_embedding = text.last_hidden_state
        pooler_output = text.pooler_output

        if self.aligned:
            audio_d2v = self.audio_model(audio_wav)
            video_res = self.video_model(video_rgb)
        else: 
            audio_d2v = self.audio_model(audio_wav, output_hidden_states=True)
            video_res = self.video_model(video_rgb)

        if self.use_xlstm:
            audio_d2v = self.audio_d2v_xlstm_proj(audio_d2v.last_hidden_state)
            # audio_feat = self.audio_feat_xlstm_proj(audio_feat).hidden_states
            # video_feat = self.video_feat_xlstm_proj(video_feat).hidden_states
        else:
            audio_d2v = self.audio_proj(audio_d2v.last_hidden_state, audio_lengths)
        audio_d2v_hs = audio_d2v.hidden_states
        video_res_hs = video_res.hidden_states

        shifted_output = self.graph_shifting_gate(
            text_embed=text_embedding, 
            visual_res=video_res_hs, 
            audio_d2v=audio_d2v_hs, 
        )
        text_shifted = shifted_output.text_shifted
        # visual_shifted = shifted_output.visual_shifted
        # audio_shifted = shifted_output.audio_shifted

        # Supervised Output
        # final_out = self.text_classifier(text_reg)
        
        # Self-Supervised Outputs
        # fusion
        # fusion_h = torch.cat([text_reg, audio_shifted, visual_shifted], dim=-1)
        fusion_h = text_shifted
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # text
        text_h = self.post_text_dropout(pooler_output)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio_d2v.pooler_output)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video_res.pooler_output)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)


        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)
        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)
        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)
        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = EasyDict({
            # 'M': final_out,
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        })
        return res
       


