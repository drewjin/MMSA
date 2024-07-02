import torch
import torch.nn as nn


__all__ = ['EnhanceNet_v1', 'EnhanceNet_v2']


class FeedForwardNetwork(nn.Module):
    def __init__(self, args, in_embed_dim, o_embed_dim) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.args = args
        self.MLP = nn.Sequential(
            nn.Linear(in_embed_dim, 2*in_embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(2*in_embed_dim, o_embed_dim),
        )
    
    def forward(self, X):
        return self.MLP(X)
    

class Decomposer(nn.Module):
    """
        Decomposer: Decompose the input into activities
    """
    def __init__(self, args, split_rate) -> None:
        super(Decomposer, self).__init__()
        self.args = args
        self.split_rate = split_rate

    def forward(self, input):
        if input.shape[-1] != sum(self.split_rate):
            raise ValueError("Input feature dimension does not match the sum of split_rate")
        decomposed_features = []
        current_index = 0
        for rate in self.split_rate:
            feature_slice = input[..., current_index:current_index + rate]
            decomposed_features.append(feature_slice)
            current_index += rate
        return decomposed_features
    
    def get_dec_heads(self):
        return len(self.split_rate)
    

class DecomposeAttention(nn.Module):
    """
        DecomposeAttention: Decomposed Features were used as input to the attention mechanism
        - Ordinary dot product attention (v1)
        - Multi-head attention (v2)
    """
    def __init__(self, args, version, split_rate) -> None:
        super(DecomposeAttention, self).__init__()
        MODEL_MAP = {'v1': self.__dot_product, 'v2': self.__mh_attn}
        MODEL_MAP[version](args, split_rate)
    
    def __dot_product(self, args, split_rate):
        nheads = args['num_specific_heads']
        if nheads != 1:
            raise NotImplementedError(
                r"The current implementation only supports nheads equal to 1." 
                r"Please adjust the value of nheads accordingly."
            )
        dropout = args['dropout']
        self.attn_layers = nn.ModuleList([ # NOTE： How to parallelize the computation?
            nn.MultiheadAttention(embed_dim=dec_size, num_heads=nheads, dropout=dropout)
            for dec_size in split_rate
        ])

    def __mh_attn(self, args, split_rate):
        nheads = args['num_specific_heads']
        dropout = args['dropout']
        self.attn_layers = nn.ModuleList([ # NOTE： How to parallelize the computation?
            nn.MultiheadAttention(embed_dim=dec_size, num_heads=nhead, dropout=dropout)
            for dec_size, nhead in zip(split_rate, nheads)
        ])

    def forward(self, X):
        return torch.cat([attn_layer_i(query=x_i, key=x_i, value=x_i)[0] 
                          for x_i, attn_layer_i in zip(X, self.attn_layers)], dim=-1)


class DecomposeEnhanceLayer(nn.Module):
    """
        DecomposeEnhanceLayer: 
    """
    def __init__(self, args, split_rate, version) -> None:
        super(DecomposeEnhanceLayer, self).__init__()
        self.args = args
        self.use_residual = args.get('residual', True)
        self.use_ffn = args.get('ffn', True)
        self.embed_dim = embed_dim = sum(split_rate)
        self.decomposer = Decomposer(args, split_rate)
        self.dec_attn   = DecomposeAttention(args['decompose_attn'], version, split_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        if self.use_ffn:
            self.ffn = FeedForwardNetwork(args['decompose_ffn'], embed_dim, embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, X):
        # Decompose Attention
        res_X = X
        dec_X = self.decomposer(X)
        dec_att_X = self.dec_attn(dec_X)
        if self.use_residual:
            H = self.layer_norm1(dec_att_X + res_X)
        else:
            H = self.layer_norm1(dec_att_X)
        
        # Feed Forward Network
        res_H = H
        if self.use_ffn:
            T = self.ffn(H)
            if self.use_residual:
                O = self.layer_norm2(T + res_H)
            else:
                O = self.layer_norm2(T)
        else:
            O = H

        return O
        


class DecomposeEnhanceBlock(nn.Module):
    """
        DecomposeEnhanceBlock:
    """
    def __init__(self, args, split_rate, version) -> None:
        super(DecomposeEnhanceBlock, self).__init__()
        self.args  = args
        num_layers = args.get('num_layers', 1)
        a_sr, v_sr = split_rate.values()
        self.v_layers = nn.Sequential(*[
            DecomposeEnhanceLayer(args, v_sr, version) 
            for _ in range(num_layers)
        ])
        self.a_layers = nn.Sequential(*[
            DecomposeEnhanceLayer(args, a_sr, version) 
            for _ in range(num_layers)
        ])

    def forward(self, X_v, X_a):
        return (self.v_layers(X_v), self.a_layers(X_a))


class BaseEnhanceLayer(nn.Module):
    """
        EarlyEnhanceLayer
    """
    def __init__(self, args, embed_dim):
        super(BaseEnhanceLayer, self).__init__()
        self.args = args
        self.use_residual = args.get('residual', True)
        self.use_ffn = args.get('ffn', True)
        attn_param = args.get('attn', None)
        ffn_param = args.get('mlp', None)
        proj_dim = attn_param['embed_dim']
        self.proj1 = nn.Linear(embed_dim, proj_dim)
        self.mh_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=attn_param['num_heads'], dropout=attn_param['dropout'], batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(proj_dim)
        if args['ffn']:
            self.ffn = FeedForwardNetwork(ffn_param, proj_dim, proj_dim)
            self.layer_norm2 = nn.LayerNorm(proj_dim)
        self.proj2 = nn.Linear(proj_dim, embed_dim)


    def forward(self, X):
        X = self.proj1(X)
        res_X = X
        X = self.mh_attn(X, X, X)[0]
        if self.use_residual:
            H = self.layer_norm1(X + res_X)
        else:
            H = self.layer_norm1(X)
        
        res_H = H
        if self.use_ffn:
            T = self.ffn(H)
            if self.use_residual:
                O = self.layer_norm2(T + res_H)
            else:
                O = self.layer_norm2(T)
        else:
            O = H
        
        return self.proj2(O)

class BaseEnhanceBlock(nn.Module):
    """
        EarlyEnhanceBlock
    """
    def __init__(self, args, split_rate):
        super(BaseEnhanceBlock, self).__init__()
        self.args = args
        num_layers = args.get('num_layers', 1)
        a_embed_dim, v_embed_dim = sum(split_rate['audio']), sum(split_rate['vision'])
        self.v_layers = nn.Sequential(*[
            BaseEnhanceLayer(args, v_embed_dim)
            for _ in range(num_layers)
        ])
        self.a_layers = nn.Sequential(*[
            BaseEnhanceLayer(args, a_embed_dim)
            for _ in range(num_layers)
        ])
    def forward(self, X_v, X_a):
        return (self.v_layers(X_v), self.a_layers(X_a))
    

class LateFFN(nn.Module):
    def __init__(self, args, v_embed_dim, a_embed_dim):
        super(LateFFN, self).__init__()
        self.v_net = nn.Sequential(
            nn.LayerNorm(v_embed_dim),
            FeedForwardNetwork(args, v_embed_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim)
        )
        self.a_net = nn.Sequential(
            nn.LayerNorm(a_embed_dim),
            FeedForwardNetwork(args, a_embed_dim, a_embed_dim),
            nn.LayerNorm(a_embed_dim)
        )

    def forward(self, X_v, X_a):
        return (self.v_net(X_v), self.a_net(X_a))


class EnhanceNet_v1(nn.Module):
    """
        EnhanceNet_v1: Using only Decomposer and DecomposeAttention(none Multi-Head version or Single-Head version)
    """
    def __init__(self, args, params) -> None:
        super(EnhanceNet_v1, self).__init__()
        split_rate   = args.get('split_rate', None)
        self.params  = params
        self.version = 'v1'

        self.v_sr   = split_rate['vision']
        self.a_sr   = split_rate['audio']

        self.v_dec  = Decomposer(args, self.v_sr)
        self.a_dec  = Decomposer(args, self.a_sr)

        self.v_dec_attn = DecomposeAttention(params, self.version, self.v_sr)
        self.a_dec_attn = DecomposeAttention(params, self.version, self.a_sr)
    
    def forward(self, vision, audio):
        split_v, split_a = self.v_dec(vision), self.a_dec(audio)
        enhanced_v, enhanced_a = self.v_dec_attn(split_v), self.a_dec_attn(split_a)
        return enhanced_v, enhanced_a


class EnhanceNet_v2(nn.Module):
    """
        EnhanceNet_v2:  
        - Decomposer
        - DecomposeAttention(Multi-Head version)
        - GlobalMultiHeadAttention and ResidualConnection
    """
    def __init__(self, args, params) -> None:
        super(EnhanceNet_v2, self).__init__()
        split_rate   = args.get('split_rate', None)
        self.params  = params
        self.version = 'v2'
        if params['use_decomposition']: 
            self.dec_en_net = DecomposeEnhanceBlock(params['decompose'], split_rate, self.version)
        if params['use_early_enhance']:
            self.early_en_net = BaseEnhanceBlock(params['early_enhance'], split_rate)
            self.late_ffns = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))
        if params['use_late_enhance']:
            self.late_en_net = BaseEnhanceBlock(params['late_enhance'], split_rate)
        if params['global_residual']:
            self.glb_ffns = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))

    def forward(self, vision, audio):
        glb_res = self.params.get('global_residual', False)
        early = self.params.get('use_early_enhance', True)
        late = self.params.get('use_late_enhance', True)
        dec = self.params.get('use_decomposition', True)

        if glb_res:
            res_org_v, res_org_a = vision, audio
        
        if dec:
            O_dec_v, O_dec_a = self.dec_en_net(vision, audio)
            if late:
                O_l_v, O_l_a = self.late_en_net(O_dec_v, O_dec_a)
                if early:
                    O_e_v, O_e_a = self.early_en_net(vision, audio) 
                    O_v, O_a = self.late_ffns(O_l_v + O_e_v, O_l_a + O_e_a)
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_v + res_org_v, O_a + res_org_a)
                else:
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_l_v + res_org_v, O_l_a + res_org_a)
                    else:
                        O_v, O_a = O_l_v, O_l_a
        else:
            if late:
                O_l_v, O_l_a = self.late_en_net(vision, audio)
                if early:
                    O_e_v, O_e_a = self.early_en_net(vision, audio) 
                    O_v, O_a = self.late_ffns(O_l_v + O_e_v, O_l_a + O_e_a)
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_v + res_org_v, O_a + res_org_a)
                else:
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_l_v + res_org_v, O_l_a + res_org_a)
                    else:
                        O_v, O_a = O_l_v, O_l_a
            else:
                O_e_v, O_e_a = self.early_en_net(vision, audio)
                if glb_res:
                    O_v, O_a = self.glb_ffns(O_e_v + res_org_v, O_e_a + res_org_a)
                else:
                    O_v, O_a = O_e_v, O_e_a
        return O_v, O_a
            
                

if __name__ == '__main__':
    def get_config_regression(
        model_name: str, dataset_name: str, config_file: str = ""
    ) -> dict:
        """
        Get the regression config of given dataset and model from config file.

        Parameters:
            model_name: Name of model.
            dataset_name: Name of dataset.
            config_file: Path to config file, if given an empty string, will use default config file.

        Returns:
            config (dict): config of the given dataset and model
        """
        import os
        import json
        from pathlib import Path
        from easydict import EasyDict as edict
        
        if config_file == "":
            config_file = Path(__file__).parent / "config" / "config_regression.json"
        with open(config_file, 'r') as f:
            config_all = json.load(f)
        model_common_args = config_all[model_name]['commonParams']
        model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
        dataset_args = config_all['datasetCommonParams'][dataset_name]
        # use aligned feature if the model requires it, otherwise use unaligned feature
        if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        else:
            dataset_args = dataset_args['unaligned']
        enhance_net_args = config_all['enhanceNetParams']
        en_version = model_common_args['enhance_net_version']
        if en_version == 1:
            enhance_net_args = {'enhance_net_args':{
                'split_rate': enhance_net_args['enhancement_split'],
                'version': 'v1',
                'hyper_params': enhance_net_args['v1']
            }}
        elif en_version == 2:
            enhance_net_args = {'enhance_net_args':{
                'split_rate': enhance_net_args['enhancement_split'],
                'version': 'v2',
                'hyper_params': enhance_net_args['v2']
            }}

        config = {}
        config['model_name'] = model_name
        config['dataset_name'] = dataset_name
        config.update(dataset_args)
        config.update(model_common_args)
        config.update(model_dataset_args)
        config.update(enhance_net_args)
        config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
        config = edict(config) # use edict for backward compatibility with MMSA v1.0

        pretrained_weight_root = config_all['pretrainedWeights']['weights_root_dir']
        if config['transformers'] not in [None, [], '']:
            config['weight_dir'] = os.path.join(pretrained_weight_root, config['transformers'], config['pretrained'])
        else:
            config['weight_dir'] = os.path.join(pretrained_weight_root, config['pretrained'])

        weight_dir = config['weight_dir']
        if config['transformers'] == 'bert':
            weight_dir = weight_dir.split('/')
            cn_bert = 'bert-base-chinese'
            en_bert = 'bert-base-uncased'
            if config['language']== 'cn' and weight_dir[-1] != cn_bert:
                weight_dir[-1] = cn_bert
                weight_dir = '/'.join(weight_dir)
                config['weight_dir'] = weight_dir 
            elif config['language'] == 'en' and weight_dir[-1] != en_bert:
                weight_dir[-1] = en_bert
                weight_dir = '/'.join(weight_dir)
                config['weight_dir'] = weight_dir 

        return config
    
    model = 'mult'
    dataset = 'sims'
    cfg = '/home/drew/Desktop/Research/MMSA/src/MMSA/config/config_regression.json'
    args = get_config_regression(model, dataset, cfg)
    
    vision, audio = torch.randn(16, 50, 709), torch.randn(16, 50, 33)
    enargs = args.get('enhance_net_args', None)
    version = enargs['version']
    params = enargs['hyper_params']
    
    ENHANCENET_MAP = {'v1': EnhanceNet_v1, 'v2': EnhanceNet_v2}
    enhanceNet = ENHANCENET_MAP[version](enargs, params)
    output = enhanceNet(vision, audio)