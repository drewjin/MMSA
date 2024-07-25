#!/bin/bash

# 定义模型列表
model_list=(
    # 'lf_dnn' 
    # 'ef_lstm' 
    # 'tfn' 'mctn' 'lmf' 
    # 'bm_mag_m'
    # 'mfn' 
    # 'graph_mfn' 
    'cmgformer'
    'self_mm' 
    'tetfn' 
    'cenet'
    'mmim'
    'mult' 
    'bert_mag'
    'misa' 
    'tfr_net'
    'mfm' 
    'mlf_dnn' 
    'mtfn' 
    'mlmf' 
)

# 执行Python脚本的路径
python_script_path="/home/drew/Desktop/Research/MMSA/src/MMSA/__main__.py"

# 遍历模型列表，依次运行实验
for model in "${model_list[@]}"
do
    echo "[Executing]: python $python_script_path -m $model -d mosei"
    
    # 执行Python脚本
    python "$python_script_path" -m "$model" -d mosei
    
    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "Error occurred while running experiment for model $model"
    fi
done