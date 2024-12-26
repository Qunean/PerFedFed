##!/bin/bash
#
## 所有方法列表
#methods=("local" "fedavg" "fedfed" "fedgen" "fedprox" "moon" "fedlc" "ccvr" "pfedsim" "fedper" "fedrep" "fedAP" "fedbn" "fedfomo" "fedproto" "perfedfed")
##methods=("perfedfed")
## 输出日志文件
#log_file="output_cifar10a1.0_lr0.003.txt"
#
## 清空日志文件
#> "$log_file"
#
## 循环每个方法并按顺序运行
#for method in "${methods[@]}"; do
#    # 循环 datasets_weights 从 0.0 到 1.0，步长为 0.1
##    for datasets_weights in $(seq 0.0 0.1 1.0); do
#        # 构造对应方法的 CSV 文件路径
##        csv_path="out/$method/mnist/datasets_weights=$datasets_weights/metrics.csv"
##        csv_path="/out/$method/cifar10/cifar10a1.0/metrics.csv"
#        # 查找 metrics.csv 文件，限制每次查找结果为唯一子目录下的文件
#        csv_path=$(find "out/$method/cifar10/" -type f -name "metrics.csv" | head -n 1)
#        # 检查 CSV 文件是否存在
#        if [[ -f "$csv_path" ]]; then
#            echo "Processing metrics for method: $method with datasets_weights: $datasets_weights" | tee -a "$log_file"
#            # 调用 Python 脚本处理 CSV 文件并将输出追加到日志
#            python maxAccinCSV.py "$csv_path" >> "$log_file" 2>&1
#        else
#            echo "CSV file not found for method: $method with datasets_weights: $datasets_weights at path: $csv_path" | tee -a "$log_file"
#            continue
#        fi
##    done
#done
#
#echo "All tasks completed." | tee -a "$log_file"

# 所有方法列表
methods=("perfedfed")

# 输出日志文件
log_file="output_cifar10a1.0_vaelr_dataweig.txt"

# 清空日志文件
> "$log_file"

# 循环每个方法并按顺序运行
for method in "${methods[@]}"; do
    # 查找 cifar10 下所有 metrics.csv 文件
    csv_files=$(find "../grid_search/out/$method/" \
        -type f -path "*/cifar10/*" -name "metrics.csv")

    # 检查是否找到文件
    if [[ -z "$csv_files" ]]; then
        echo "No metrics.csv files found for method: $method in cifar10 subdirectories." | tee -a "$log_file"
        continue
    fi

    # 遍历所有找到的 CSV 文件
    for csv_path in $csv_files; do
        if [[ -f "$csv_path" ]]; then
            echo "Processing metrics for method: $method at path: $csv_path" | tee -a "$log_file"
            python maxAccinCSV.py "$csv_path" >> "$log_file" 2>&1
        else
            echo "File not found (unexpected error) at path: $csv_path" | tee -a "$log_file"
        fi
    done
done

echo "All tasks completed." | tee -a "$log_file"
