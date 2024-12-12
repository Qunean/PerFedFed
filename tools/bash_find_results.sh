#!/bin/bash

# 所有方法列表
methods=("local" "fedavg" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedper" "fedrep" "fedAP" "fedbn" "fedfomo" "fedproto" "perfedfed")

# 输出日志文件
log_file="output_cifar10a0.1_maxacc.txt"

# 清空日志文件
> "$log_file"

# 循环每个方法并按顺序运行
for method in "${methods[@]}"; do
    # 构造对应方法的 CSV 文件路径
    csv_path="../out/$method/cifar10/cifar10a0.1/metrics.csv"

    # 检查 CSV 文件是否存在
    if [[ -f "$csv_path" ]]; then
        echo "Processing metrics for method: $method" | tee -a "$log_file"
        # 调用 Python 脚本处理 CSV 文件并将输出追加到日志
        python maxAccinCSV.py "$csv_path" >> "$log_file" 2>&1
    else
        echo "CSV file not found for method: $method at path: $csv_path" | tee -a "$log_file"
        continue
    fi
done

echo "All tasks completed." | tee -a "$log_file"
