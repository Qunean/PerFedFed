#!/bin/bash

# 所有方法列表
methods=("local" "fedavg" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedper" "fedrep" "fedAP" "fedbn" "fedfomo" "fedproto" "perfedfed")
#methods=("local" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedper" "fedrep" "fedAP" "fedbn" "fedfomo" "fedproto")
global_buffer_list=("local" "fedavg" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedAP" "fedbn" "fedfomo" "fedproto")
local_buffer_list=("fedper" "fedrep" "perfedfed")

# 配置文件路径
config_path="config/cifar10/a=0.5"

# 输出日志文件
log_file="output_cifar10a0.5.txt"

# 清空日志文件
> "$log_file"

# 循环每个方法并按顺序运行
for method in "${methods[@]}"; do
    if [[ " ${global_buffer_list[@]} " =~ " $method " ]]; then
        config_name="buffer_global"
    elif [[ " ${local_buffer_list[@]} " =~ " $method " ]]; then
        config_name="buffer_local"
    else
        echo "Unknown method: $method" | tee -a "$log_file"
        continue
    fi

    echo "Running method: $method with config: $config_name" | tee -a "$log_file"

    # 运行每个方法，日志输出保存到文件
    if python main.py --config-name "$config_name" --config-path "$config_path" method="$method" >> "$log_file" 2>&1; then
        echo "Method $method completed successfully." | tee -a "$log_file"
    else
        echo "Method $method failed." | tee -a "$log_file"
    fi
done

echo "All tasks completed." | tee -a "$log_file"