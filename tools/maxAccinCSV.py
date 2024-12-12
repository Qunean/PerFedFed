import pandas as pd
import argparse

def process_csv(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 找到 accuracy_val_before 最大值时的行
    max_val_before_row = df[df['accuracy_val_before'] == df['accuracy_val_before'].max()]

    # 获取对应的 accuracy_test_before 和 accuracy_test_after 的值
    accuracy_test_before = max_val_before_row['accuracy_test_before'].iloc[0]
    accuracy_test_after = max_val_before_row['accuracy_test_after'].iloc[0]
    epoch_max_val = max_val_before_row['epoch'].iloc[0]

    # 找到最高的 accuracy_test_after
    max_test_after_row = df[df['accuracy_test_after'] == df['accuracy_test_after'].max()]
    highest_accuracy_test_after = max_test_after_row['accuracy_test_after'].iloc[0]
    epoch_max_test_after = max_test_after_row['epoch'].iloc[0]

    # 输出结果
    print(f"accuracy_test_before at max accuracy_val_before: {accuracy_test_before:.2f}% ({epoch_max_val})")
    print(f"accuracy_test_after at max accuracy_val_before: {accuracy_test_after:.2f}% ({epoch_max_val})")
    print(f"Highest accuracy_test_after: {highest_accuracy_test_after:.2f}% ({epoch_max_test_after})")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process a CSV file and extract specific metrics.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理函数
    process_csv(args.file_path)
