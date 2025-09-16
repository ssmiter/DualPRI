import os
import numpy as np
import pickle
from tqdm import tqdm

def parse_pssm_to_array(pssm_file):
    """
    解析PSSM文件为数组，支持不同格式的PSSM文件
    """
    with open(pssm_file, 'r') as file:
        lines = file.readlines()

    pssm_data = []
    matrix_started = False

    for line in lines:
        # 跳过空行
        if line.strip() == "":
            continue

        # 检查是否包含标准PSSM矩阵开始标记
        if "Last position-specific scoring matrix" in line:
            matrix_started = True
            continue

        # 如果找到了标记，跳过接下来的2行标题
        if matrix_started:
            matrix_started = False
            continue

        # 尝试解析行数据
        parts = line.strip().split()

        # 确保行包含足够的数据
        if len(parts) < 22:  # PSSM矩阵每行应该至少有22个值
            continue

        try:
            # 尝试提取数值部分（前20个得分）
            scores = []
            for value in parts[2:22]:  # 跳过前两个列（序号和氨基酸）
                # 跳过非数字值
                if not value.replace('-', '').isdigit():
                    continue
                scores.append(int(value))

            if len(scores) == 20:  # 确保获取了完整的20个得分
                pssm_data.append(scores)

        except (ValueError, IndexError):
            continue

    # 如果没有获取到有效数据，尝试直接解析第一组20列数据
    if not pssm_data:
        for line in lines:
            if line.strip() and not line.startswith('#'):  # 跳过注释和空行
                parts = line.strip().split()
                try:
                    scores = [int(x) for x in parts[:20]]  # 仅取前20个数值
                    if len(scores) == 20:
                        pssm_data.append(scores)
                except (ValueError, IndexError):
                    continue

    if not pssm_data:
        raise ValueError(f"Unable to parse PSSM data from file: {pssm_file}")

    return np.array(pssm_data)


def process_pssm_files(directory):
    """处理目录中的所有PSSM文件"""
    pssm_arrays = {}
    errors = []

    total = len(directory)
    with tqdm(total=total, desc="Processing pssm files", ncols=100) as pbar:
        for filename in os.listdir(directory):
            if filename.endswith(".pssm"):
                file_path = os.path.join(directory, filename)
                try:
                    pssm_array = parse_pssm_to_array(file_path)
                    pssm_arrays[filename] = pssm_array

                    # 更新进度条描述
                    pbar.set_postfix({
                        'Current': f"{filename}"
                    })
                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")
                    continue
                finally:
                    pbar.update(1)

    # 保存成功解析的数据
    with open(os.path.join(directory, f'pssm_s{dataset}.pkl'), 'wb') as pickle_file:
        pickle.dump(pssm_arrays, pickle_file)

    print(f"Successfully processed {len(pssm_arrays)} PSSM files")
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

dataset = '394'
# dataset = 'myoglobin'

# Replace with your directory containing .pssm files
pssm_directory = f'./Dataset/PSSM_{dataset}'
process_pssm_files(pssm_directory)
