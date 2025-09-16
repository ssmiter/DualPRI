"""dataset_process/config.py"""
# config.py
from pathlib import Path

# 全局
NUM_HOPS = 4
ENCODING_DIM = 16

class Config:
    # 数据路径
    DATA_ROOT = Path("Dataset")
    PDB_DIR = DATA_ROOT / "S2648"
    PSSM_FILE = DATA_ROOT / "pssm_s2648.pkl"
    CONSERVATION_DIR = DATA_ROOT / "cons_s2648"
    OUTPUT_DIR = Path("./pkl")

    # 图构建参数
    CONTACT_CUTOFF = 8.0  # Å

    # 特征维度
    NODE_FEATURE_DIM = 53
    EDGE_FEATURE_DIM = 16

    # 数据集信息
    DDG_FILE = DATA_ROOT / "S2648.csv"  # DDG值文件

# 创建输出目录
Config.OUTPUT_DIR.mkdir(exist_ok=True)


# config/feature_config.py


class FeatureConfig:
    """特征维度配置类"""
    # 节点特征维度 (总计67维)
    ONE_HOT_DIM = 20      # One-hot编码
    SEQ_PROPS_DIM = 12    # 序列属性特征(疏水性、体积、极性等)
    # 暂时不使用结构特征
    STRUCTURE_DIM = 14    # 结构特征(二级结构3维、ASA/RSA 2维、B-factor 1维、局部环境8维)

    EVOLUTION_DIM = 21    # 进化特征(PSSM 20维 + conservation score 1维)

    # 边特征维度 (总计16维)
    DISTANCE_FEATURES_DIM = 4    # 距离相关特征
    SEQUENCE_FEATURES_DIM = 4    # 序列分离特征
    CONTACT_FEATURES_DIM = 4     # 接触特征
    RESIDUE_FEATURES_DIM = 4     # 残基对特征

    @classmethod
    def get_total_node_dim(cls):
        """获取总节点特征维度"""
        return cls.ONE_HOT_DIM + cls.SEQ_PROPS_DIM + cls.STRUCTURE_DIM + cls.EVOLUTION_DIM

    @classmethod
    def get_total_edge_dim(cls):
        """获取总边特征维度"""
        return (cls.DISTANCE_FEATURES_DIM + cls.SEQUENCE_FEATURES_DIM +
                cls.CONTACT_FEATURES_DIM + cls.RESIDUE_FEATURES_DIM)

    @classmethod
    def get_node_feature_slices(cls):
        """获取节点特征的切片位置"""
        start = 0
        slices = {}

        # One-hot编码 (20维)
        slices['one_hot'] = slice(start, start + cls.ONE_HOT_DIM)
        start += cls.ONE_HOT_DIM

        # 序列属性 (12维)
        slices['seq_properties'] = slice(start, start + cls.SEQ_PROPS_DIM)
        start += cls.SEQ_PROPS_DIM

        # 结构特征 (14维)
        # 暂时不使用结构特征
        # slices['structure'] = slice(start, start + cls.STRUCTURE_DIM)
        # start += cls.STRUCTURE_DIM

        # 进化特征 (21维)
        slices['evolution'] = slice(start, start + cls.EVOLUTION_DIM)

        return slices

    @classmethod
    def get_edge_feature_slices(cls):
        """获取边特征的切片位置"""
        start = 0
        slices = {}

        # 距离特征 (4维)
        slices['distance'] = slice(start, start + cls.DISTANCE_FEATURES_DIM)
        start += cls.DISTANCE_FEATURES_DIM

        # 序列分离特征 (4维)
        slices['sequence'] = slice(start, start + cls.SEQUENCE_FEATURES_DIM)
        start += cls.SEQUENCE_FEATURES_DIM

        # 接触特征 (4维)
        slices['contact'] = slice(start, start + cls.CONTACT_FEATURES_DIM)
        start += cls.CONTACT_FEATURES_DIM

        # 残基对特征 (4维)
        slices['residue'] = slice(start, start + cls.RESIDUE_FEATURES_DIM)

        return slices

    @classmethod
    def verify_dimensions(cls, features):
        """验证特征维度是否正确"""
        expected_dim = cls.get_total_node_dim()
        actual_dim = features.shape[-1]
        if actual_dim != expected_dim:
            raise ValueError(f"Feature dimension mismatch! Expected {expected_dim} but got {actual_dim}")
        return True