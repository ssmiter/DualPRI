# config.py
import os
import sys


def get_config(node_features):
    gmb_args = {
        'd_model': 64,  # 应与 hidden_channels 匹配
        'd_state': 16,
        'd_conv': 2,
        'expand': 1,
        'use_checkpointing': True
    }
    model_args = {
        'in_channels': node_features,
        'hidden_channels': 64,
        'out_channels': 1,
        'gmb_args': gmb_args,
        'num_layers': 3
    }

    return model_args


# 全局配置: 默认
ENCODING_DIM = 16
BATCH_SIZE = 16        # 可以适当调整batch size，因为每个样本现在使用突变局部子图
LEARNING_RATE = 0.001   # 可以适当调整学习率
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100        # 可以适当调整, 建议全图300，子图100
NUM_LAYERS = 3
SEED = 42  # 41, 42
SHUFFLE = True
VAL_SPLIT = 0.1
DEFAULT_CONTACT_THRESHOLDS = [8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
# DEFAULT_CONTACT_THRESHOLDS = [7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 30.0, 40.0, 50.0]
DEFAULT_CHUNK_SIZE = 32  # 32

# 模型参数


# 子图相关配置
USE_SUBGRAPHS = True    # 为False时，使用原始图数据，NUM_HOPS不起作用
NUM_HOPS = 3            # k-hop邻居数（3, 4, 5, 6）, 推荐（3, 4）

# 其他
NUM_WORKERS = 0
PREFETCH_FACTOR = 1
DEFAULT_ESM2_CACHE_DIR = "./esm2_features"
ESM2_FEATURE_DIM = 1280
ESM2_FEATURE_TYPES = [4, 5, 6, 7]

# 项目根目录
DIR="/media/ST-18T/cheery/PRITrans"
# 常用目录
DATA_DIR = f"{DIR}/dataset_process/pkl"
MODEL_DIR = f"{DIR}/model_zoo"
# LOG_DIR = f"{DIR}/logs"


def import_from_path(file_path, module_name=None):
    """从文件路径导入模块"""
    import importlib.util

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模块文件不存在: {file_path}")

    if module_name is None:
        module_name = os.path.basename(file_path).split('.')[0]

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportError(f"导入模块 {file_path} 失败: {str(e)}")