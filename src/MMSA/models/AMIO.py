"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .multiTask import *
from .singleTask import *
from .missingTask import *
from .subNets import AlignSubNet
from .subNets import EnhanceNet_v1, EnhanceNet_v2, EnhanceNet_v3
from pytorch_transformers import BertConfig

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.args = args
        self.MODEL_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mctn': MCTN,
            'bert_mag': BERT_MAG,
            'mult': MULT,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            'cenet': CENET,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'tetfn': TETFN,
            # missing-task
            'tfr_net': TFR_NET
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]

        self.need_data_enhancement = args.get('need_data_enhancement', None)
        if self.need_data_enhancement:
            enargs = args.get('enhance_net_args', None)
            version = enargs['version']
            params = enargs['hyper_params']
            ENHANCENET_MAP = {'v1': EnhanceNet_v1, 'v2': EnhanceNet_v2, 'v3': EnhanceNet_v3}
            self.enhanceNet = ENHANCENET_MAP[version](enargs, params)
        

        if args.model_name == 'cenet':
            config = BertConfig.from_pretrained(args.weight_dir, num_labels=1, finetuning_task='sst')
            self.Model = CENET.from_pretrained(args.weight_dir, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True, args=args)
        else:
            self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        if self.need_data_enhancement:
            if self.args['model_name'] in ['self_mm', 'tetfn', 'tfr_net']:
                video_ex, audio_ex = self.enhanceNet(video_x[0], audio_x[0])
                video_x, audio_x = tuple([video_ex] + list(video_x[1:])), tuple([audio_ex] + list(audio_x[1:]))
            else:
                video_x, audio_x = self.enhanceNet(video_x, audio_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
