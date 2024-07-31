import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Data2VecAudioModel
from torch.nn.utils.rnn import pack_padded_sequence

from ..subNets import BertTextEncoder
from .modules import TempResNet
from .modules import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig

__all__ = ['GSG']

class GSG(nn.Module):
    def __init__(self, args):
        super(GSG, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(
            use_finetune=args.use_finetune, 
            transformers=args.transformers, 
            pretrained=args.weight_dir
        )

        # audio-vision subnets
        # audio_in, video_in = args.feature_dims[1:]
        # self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
        #                     num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        # self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
        #                     num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        self.audio_model = Data2VecAudioModel.from_pretrained(args.data2vec)
        self.audio_xlstm_proj = xLSTMProj(args)
        self.video_model = TempResNet(args)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        # self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        # self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        # self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim//2)
        # self.post_text_layer_3 = nn.Linear(args.post_text_dim//2, 1)

        
        # AUDIO
        # the classify layer for raw audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim//2)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim//2, 1)
        # the classify layer for audio features
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_feature_layer_1 = nn.Linear(args.audio_feature_dim, args.post_audio_feature_dim)
        self.post_audio_feature_layer_2 = nn.Linear(args.post_audio_feature_dim, args.post_audio_feature_dim)
        self.post_audio_feature_layer_3 = nn.Linear(args.post_audio_feature_dim, 1)
        # the classify layer for audio fusion
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out + args.audio_feature_dim, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim//2)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim//2, 1)

        
        # VIDEO
        # the classify layer for raw video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim//2)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim//2, 1)
        # the classify layer for video features
        self.post_audio_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_audio_feature_layer_1 = nn.Linear(args.video_feature_dim, args.post_video_feature_dim)
        self.post_audio_feature_layer_2 = nn.Linear(args.post_video_feature_dim, args.post_video_feature_dim)
        self.post_audio_feature_layer_3 = nn.Linear(args.post_video_feature_dim, 1)
        # the classify layer for video fusion
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out + args.video_feature_dim, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim//2)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim//2, 1)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio)
            video = self.video_model(video)
        else: 
            audio = self.audio_model(audio, output_hidden_states=True)
            video = self.video_model(video)
        audio = self.audio_xlstm_proj(audio.last_hidden_state)
        video = video.pooler_output
        

        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
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


        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res
    

class xLSTMProj(nn.Module):
    def __init__(self, args):
        super(xLSTMProj, self).__init__()
        self.xlstm = xLSTMBlockStack(self.xlstm_cfg(args))
        self.dropout = nn.Dropout(args.a_lstm_dropout)
        self.proj = nn.Linear(args.audio_embed, args.audio_out)

    def xlstm_cfg(self, args):
        return xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(), 
            context_length=46, 
            num_blocks=8, 
            embedding_dim=args.audio_embed, 
        )
    
    def forward(self, x):
        hidden_states = self.xlstm(x)
        hidden_states = self.dropout(hidden_states)
        last_hidden = hidden_states.permute(1, 0, 2)[-1]
        proj_output = self.proj(last_hidden)
        return proj_output


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(
            in_size, hidden_size, num_layers=num_layers, 
            dropout=dropout, bidirectional=bidirectional, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1
