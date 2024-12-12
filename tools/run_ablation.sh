#!/bin/bash

# 所有方法列表
methods=("perfedfed")

global_buffer_list=("local" "fedavg" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedAP" "fedbn" "fedfomo" "fedproto")
local_buffer_list=("fedper" "fedrep" "perfedfed")

# 配置文件路径
config_path="config/mnist/a=0.1"

# 输出日志文件
log_file="output(mnist)_perfedfed_datasetsweights.txt"

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

    for weights in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo "Running method: $method with config: $config_name and weights: $weights" | tee -a "$log_file"

        # 运行每个方法，日志输出保存到文件
        if python ../main.py --config-name "$config_name" --config-path "$config_path" method="$method" +perfedfed.datasets_weights="$weights" >> "$log_file" 2>&1; then
            echo "Method $method with weights $weights completed successfully." | tee -a "$log_file"
        else
            echo "Method $method with weights $weights failed." | tee -a "$log_file"
        fi
    done
done

echo "All tasks completed." | tee -a "$log_file"
