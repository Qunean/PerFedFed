#!/bin/bash

# 定义超参数列表
VAE_lr=("0.0001" "0.001" "0.01")
VAE_weight_decay=("1e-6" "1e-5" "1e-4")
VAE_alpha=("1.0" "2.0" "3.0" "4.0" "5.0")
VAE_re=("1.0" "2.0" "3.0" "4.0" "5.0")
VAE_kl=("0.001" "0.01" "0.1")
VAE_ce=("1.0" "2.0" "3.0" "4.0" "5.0")
VAE_x_ce=("1.0" "2.0" "3.0" "4.0" "5.0")
consis=("1.0" "2.0" "3.0" "4.0" "5.0")
robust_consis=("1.0" "2.0" "3.0" "4.0" "5.0")
#datasets_weights=("0.5" "0.7" "0.9" "1.0")

# 输出文件夹
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

# 设置最大运行轮数
MAX_RUNS=50
run_count=0

# 暴力搜索所有超参数组合
for lr in "${VAE_lr[@]}"; do
  for wd in "${VAE_weight_decay[@]}"; do
    for alpha in "${VAE_alpha[@]}"; do
      for re in "${VAE_re[@]}"; do
        for kl in "${VAE_kl[@]}"; do
          for ce in "${VAE_ce[@]}"; do
            for x_ce in "${VAE_x_ce[@]}"; do
              for c in "${consis[@]}"; do
                for rc in "${robust_consis[@]}"; do
                  # 检查是否达到最大运行次数
                  if [ $run_count -ge $MAX_RUNS ]; then
                    echo "达到最大搜索轮数 ($MAX_RUNS)，退出循环。"
                    exit 0
                  fi

                  # 构造日志文件名
                  LOG_FILE="$OUTPUT_DIR/lr_${lr}_wd_${wd}_alpha_${alpha}_re_${re}_kl_${kl}_ce_${ce}_x_ce_${x_ce}_consis_${c}_robust_${rc}.log"

                  # 如果日志文件已存在，则跳过
                  if [ -f "$LOG_FILE" ]; then
                    echo "参数组合的结果已存在，跳过：$LOG_FILE"
                    continue
                  fi

                  # 构造命令行参数
                  PARAMS="perfedfed.VAE_lr=$lr perfedfed.VAE_weight_decay=$wd perfedfed.VAE_alpha=$alpha perfedfed.VAE_re=$re perfedfed.VAE_kl=$kl perfedfed.VAE_ce=$ce perfedfed.VAE_x_ce=$x_ce perfedfed.consis=$c perfedfed.robust_consis=$rc"

                  # 打印当前参数组合
                  echo "Running with parameters: $PARAMS"

                  # 运行主程序，并将输出保存到文件
                  python ../main.py --config-name perfedfed method=perfedfed $PARAMS > "$LOG_FILE"

                  # 增加计数器
                  run_count=$((run_count + 1))
                done
              done
            done
          done
        done
      done
    done
  done
done
