import torch

from os.path import join
from transformers import AutoModelForCausalLM, AwqConfig

weight_root = '/home/drew/Desktop/Research/weights'
weight = 'mamba/mamba-130m-hf'
target = 'mamba/mamba-130m-hf-q'
weight_path = join(weight_root, weight)
target_path = join(weight_root, target)

awq_cfg = AwqConfig(bits=4,)
quantized_model = AutoModelForCausalLM.from_pretrained(
    weight_path, device_map="auto", 
    quantization_config=AwqConfig)