"""
蛋白质-RNA相互作用模型工厂
统一的模型创建接口，适用于所有模型
"""
import os
import importlib
import torch
import torch.nn as nn
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CONTACT_THRESHOLDS


class ModelFactory:
    """
    模型工厂类，用于创建和管理不同类型的蛋白质-RNA相互作用模型
    """
    
    # 内置模型映射
    BUILTIN_MODELS = {
        'transformer': 'model_zoo.transformer.TransformerModel',
        'dualssd': 'model.DualSSD.DualSSD',  # DualSSD主模型
         # 新增对比模型
        'gnn': 'model_zoo.unified_gnn.UnifiedGNN',
        'gcn': 'model_zoo.unified_gnn.UnifiedGNN',  # 默认使用GCN
        'gat': 'model_zoo.unified_gnn.UnifiedGNN',  # 使用GAT
        'sage': 'model_zoo.unified_gnn.UnifiedGNN',  # 使用GraphSAGE
        'gin': 'model_zoo.unified_gnn.UnifiedGNN',  # 使用GIN
        'edge': 'model_zoo.unified_gnn.UnifiedGNN',  # 使用EdgeCNN
        'graph_transformer': 'model_zoo.GraphTransformer.GraphTransformer_WithRandomWalkPE',
    }

    # 模型参数映射：定义每个模型期望的参数格式
    MODEL_PARAMS_MAP = {
        # 新添加的映射
        # 在MODEL_PARAMS_MAP中添加,
        'model_zoo.transformer.TransformerModel': {
            'protein_channels': 'protein_channels',
            'rna_channels': 'rna_channels',  # 为接口兼容性保留
            'hidden_channels': 'hidden_channels',
            'out_channels': 'out_channels',
            'num_layers': 'num_layers',
            'nhead': 8,  # 默认头数
            'dropout': 'dropout',
            'dim_feedforward': 'hidden_channels * 4'  # 前馈网络维度为隐藏层的4倍
        },
        'model_zoo.unified_gnn.UnifiedGNN': {
            'protein_channels': 'protein_channels',
            'rna_channels': 'rna_channels',
            'hidden_channels': 'hidden_channels',
            'out_channels': 'out_channels',
            'num_layers': 'num_layers',
            'dropout': 'dropout',
            'pool_type': 'mean',  # 默认使用平均池化
            'norm': None,  # 默认不使用归一化
            'heads': 4,  # GAT默认头数
            'edge_dim': 16  # 边特征默认维度
            # 注意：gnn_type会在_map_params方法中通过特殊逻辑设置
        },
        'model_zoo.GraphTransformer.GraphTransformer_WithRandomWalkPE': {
            'protein_channels': 'protein_channels',
            'rna_channels': 'rna_channels',  # 为接口兼容性保留，但实际不使用
            'hidden_channels': 'hidden_channels',
            'out_channels': 'out_channels',
            'num_layers': 'num_layers',
            'dropout': 'dropout',
            'walk_length': 16,  # 默认随机游走步长
            'edge_dim': 16,  # 边特征维度，与positional_encoding函数输出一致
            'use_performer': True
        },
        'model.DualSSD.DualSSD': {
            'protein_channels': 'protein_channels',
            'rna_channels': 'rna_channels',
            'hidden_channels': 'hidden_channels',
            'out_channels': 1,
            'num_layers': 3,
            'dropout': 'dropout',
            'd_state': 32,
            'd_conv': 4,
            'expand': 2,
            'headdim': 16,
            'chunk_size': DEFAULT_CHUNK_SIZE,
            # 'num_contact_classes': num_contact_classes,
            'aux_weight': 0.2  # 使用与之前相同的辅助任务权重
        },  # it works well with Parameters(32 4 2 16 32 7 0.2)
    }

    @classmethod
    def create_model(cls, model_name, **kwargs):
        """
        创建模型实例

        参数:
        - model_name: 模型名称（预设名称或完整类路径）
        - **kwargs: 传递给模型构造函数的参数

        返回:
        - model: 模型实例
        """
        try:
            # 首先检查是否是预设模型
            if model_name in cls.BUILTIN_MODELS:
                model_path = cls.BUILTIN_MODELS[model_name]
                return cls._create_from_path(model_path, **kwargs)

            # 检查model_name是否是直接的类路径
            if '.' in model_name:
                return cls._create_from_path(model_name, **kwargs)

            # 尝试从model目录加载
            if os.path.exists(f"model/{model_name}.py"):
                return cls._create_from_path(f"model.{model_name}", **kwargs)

            # 尝试从model_zoo目录加载
            if os.path.exists(f"model_zoo/{model_name}.py"):
                return cls._create_from_path(f"model_zoo.{model_name}", **kwargs)

            # 找不到模型
            raise ValueError(f"未找到模型: {model_name}")

        except Exception as e:
            raise ValueError(f"创建模型 '{model_name}' 失败: {str(e)}")

    @classmethod
    def _create_from_path(cls, model_path, **kwargs):
        """
        从模块路径创建模型

        参数:
        - model_path: 模型类的完整路径 (例如 'model.my_model.MyModel')
        - **kwargs: 传递给模型构造函数的参数

        返回:
        - model: 模型实例
        """
        try:
            # 解析模块路径和类名
            module_path, class_name = model_path.rsplit('.', 1)

            # 导入模块
            module = importlib.import_module(module_path)

            # 获取类
            model_class = getattr(module, class_name)

            # 添加模型名称，用于统一GNN模型设置gnn_type
            if 'model' in kwargs and model_path == 'model.unified_gnn.UnifiedGNN':
                kwargs['model_name'] = kwargs['model']

            # 检查是否有create_model静态方法
            if hasattr(model_class, 'create_model'):
                return model_class.create_model(**kwargs)

            # 获取正确的参数映射
            model_params = cls._map_params(model_path, **kwargs)

            # 实例化模型
            return model_class(**model_params)

        except (ImportError, AttributeError) as e:
            raise ValueError(f"无法导入模型 '{model_path}': {str(e)}")
        except Exception as e:
            raise ValueError(f"创建模型实例失败: {str(e)}")


    @classmethod
    def _map_params(cls, model_path, **kwargs):
        """
        根据模型类型映射参数

        参数:
        - model_path: 模型类的完整路径
        - **kwargs: 原始参数

        返回:
        - 映射后的参数字典
        """
        # 如果模型有特定的参数映射配置，使用它
        if model_path in cls.MODEL_PARAMS_MAP:
            param_map = cls.MODEL_PARAMS_MAP[model_path]
            model_params = {}

            # 处理每个参数映射
            for target_param, source_param in param_map.items():
                if isinstance(source_param, dict):
                    # 处理嵌套参数如gmb_args
                    nested_params = {}
                    for nested_target, nested_source in source_param.items():
                        if isinstance(nested_source, str):
                            if nested_source in kwargs:
                                nested_params[nested_target] = kwargs[nested_source]
                                print(f"嵌套参数映射: {nested_target} = {kwargs[nested_source]} (来自 {nested_source})")
                            else:
                                print(f"警告: 嵌套参数 {nested_target} 的映射源 {nested_source} 不在用户参数中")
                        else:
                            nested_params[nested_target] = nested_source
                            print(f"嵌套参数使用直接值: {nested_target} = {nested_source}")
                    model_params[target_param] = nested_params
                elif isinstance(source_param, str):
                    # 特殊处理：参数名称相同的情况
                    if source_param == target_param and target_param in kwargs:
                        model_params[target_param] = kwargs[target_param]
                        print(f"特殊处理相同名称参数: {target_param} = {kwargs[target_param]}")
                    # 正常映射处理
                    elif source_param in kwargs:
                        model_params[target_param] = kwargs[source_param]
                        print(f"参数映射: {target_param} = {kwargs[source_param]} (来自 {source_param})")
                    else:
                        print(f"警告: 参数 {target_param} 的映射源 {source_param} 不在用户参数中")
                else:
                    # 直接值（非字符串）
                    model_params[target_param] = source_param
                    print(f"使用参数映射中的直接值: {target_param} = {source_param}")


            # 确保关键参数存在（避免None值或缺失）
            model_params.setdefault('out_channels', 1)

            print(f"模型 {model_path} 的最终参数:")
            for k, v in model_params.items():
                print(f"  {k}: {v}")

            return model_params

        # 如果没有特定配置，直接返回原始参数
        return kwargs

    @classmethod
    def get_available_models(cls):
        """
        获取所有可用模型列表

        返回:
        - 可用模型列表
        """
        available_models = list(cls.BUILTIN_MODELS.keys())

        # 扫描model目录
        if os.path.exists("model"):
            for filename in os.listdir("model"):
                if filename.endswith(".py") and not filename.startswith("__"):
                    model_name = filename[:-3]
                    if model_name not in available_models:
                        available_models.append(model_name)

        # 扫描model_zoo目录
        if os.path.exists("model_zoo"):
            for filename in os.listdir("model_zoo"):
                if filename.endswith(".py") and not filename.startswith("__"):
                    model_name = filename[:-3]
                    available_models.append(f"zoo:{model_name}")

        return available_models

    @staticmethod
    def load_model(model_path, model_class=None, **kwargs):
        """
        从保存的检查点加载模型

        参数:
        - model_path: 模型检查点路径
        - model_class: 模型类（若为None，则从检查点获取）
        - **kwargs: 传递给模型构造函数的参数

        返回:
        - 加载的模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')

        # 如果提供了模型类，直接使用
        if model_class is not None:
            model = model_class(**kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model

        # 从检查点获取模型类和参数
        if 'model_info' in checkpoint:
            model_info = checkpoint['model_info']
            model_name = model_info.get('name')
            model_module = model_info.get('module')

            if model_name and model_module:
                model_path = f"{model_module}.{model_name}"
                model = ModelFactory._create_from_path(model_path, **kwargs)
                model.load_state_dict(checkpoint['model_state_dict'])
                return model

        # 如果没有模型信息，尝试从文件名推断
        model_dir = os.path.dirname(model_path)

        # 尝试查找model_info.txt或model_info.json文件
        info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                model_name = model_info.get('name')
                model_module = model_info.get('module')

                if model_name and model_module:
                    model_path = f"{model_module}.{model_name}"
                    model = ModelFactory._create_from_path(model_path, **kwargs)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    return model

        # 尝试查找model_info.txt文件
        info_path = os.path.join(model_dir, 'model_info.txt')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                lines = f.readlines()
                model_name = None
                model_module = None

                for line in lines:
                    if line.startswith('name:'):
                        model_name = line.split(':', 1)[1].strip()
                    elif line.startswith('module:'):
                        model_module = line.split(':', 1)[1].strip()

                if model_name and model_module:
                    model_path = f"{model_module}.{model_name}"
                    model = ModelFactory._create_from_path(model_path, **kwargs)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    return model
        
        raise ValueError("无法确定模型类型。请提供model_class参数或确保检查点包含模型信息。")


def create_model(model_name, **kwargs):
    """
    创建模型的便捷函数
    
    参数:
    - model_name: 模型名称
    - **kwargs: 传递给模型构造函数的参数
    
    返回:
    - model: 模型实例
    """
    return ModelFactory.create_model(model_name, **kwargs)


def load_model(model_path, model_class=None, **kwargs):
    """
    加载模型的便捷函数
    
    参数:
    - model_path: 模型检查点路径
    - model_class: 模型类（若为None，则从检查点获取）
    - **kwargs: 传递给模型构造函数的参数
    
    返回:
    - model: 加载的模型
    """
    return ModelFactory.load_model(model_path, model_class, **kwargs)


def get_available_models():
    """
    获取所有可用模型的便捷函数
    
    返回:
    - 可用模型列表
    """
    return ModelFactory.get_available_models()