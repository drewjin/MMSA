import json
import subprocess

# 定义模型列表
model_list = [
    'lf_dnn', 'ef_lstm', 'tfn', 'mctn', 'lmf', 'mfn', 'graph_mfn', 'mult', 'bert_mag',
    'misa', 'mfm', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm', 'mmim', 'tfr_net', 'tetfn', 'cenet'
]

# 定义增强配置列表，直接使用Python列表
enhance_list = [
    [0,1], 
    [1,1],
    [1,2],
    [1,3]
]

# 定义一个函数来运行单个模型的实验
def run_experiment(model, enhance):
    # 将增强配置列表转换为JSON格式的字符串
    enhance_arg = json.dumps(enhance)

    # 构建命令行参数列表
    cmd = [
        'python', '/home/drew/Desktop/Research/MMSA/src/MMSA/__main__.py',
        '-m', model,
        '-d', 'sims',  # 确保数据集参数后面有值
        '-E', enhance_arg  # 将增强配置列表转换为JSON字符串
    ]

    # 打印将要执行的命令
    print(f"Executing: {' '.join(cmd)}")

    # 使用 subprocess 运行命令，并等待命令执行完成
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 打印输出和错误
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    # 检查命令是否成功执行
    if result.returncode != 0:
        print(f"Error occurred while running experiment for model {model} with enhance {enhance}")

# 遍历模型和增强配置列表，依次运行实验
# for model in ['mult','bert_mag','misa','self_mm','mmim','tetfn']:
for model in model_list:
    for enhance in enhance_list:
        run_experiment(model, enhance)