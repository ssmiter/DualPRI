"""
蛋白质-RNA相互作用模型训练器
统一的训练接口，适用于所有模型
"""
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# matplotlib.use('Agg')  # 设置为非交互式后端
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ProteinRNATrainer:
    """蛋白质-RNA相互作用模型训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        learning_rate=0.0001,
        weight_decay=1e-5,
        device=None,
        output_dir='./output',
        max_epochs=150,
        patience=20,
        checkpoint_interval=10,
        resume_from=None
    ):
        """
        初始化训练器
        
        参数:
        - model: 模型
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - test_loader: 测试数据加载器
        - learning_rate: 学习率
        - weight_decay: 权重衰减
        - device: 设备(若为None则自动选择)
        - output_dir: 输出目录
        - max_epochs: 最大训练轮数
        - patience: 早停耐心值
        - checkpoint_interval: 检查点保存间隔
        - resume_from: 恢复训练的检查点路径
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 模型
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8,
            patience=15,
            verbose=True
        )
        
        # 训练设置
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_interval = checkpoint_interval
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # 训练状态
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_pcc = -float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 最佳模型路径
        self.best_model_path = os.path.join(output_dir, 'best_model.pt')
        
        # 恢复训练
        if resume_from is not None:
            self.resume_training(resume_from)
    
    def resume_training(self, checkpoint_path):
        """
        从检查点恢复训练
        
        参数:
        - checkpoint_path: 检查点路径
        """
        print(f"从检查点恢复训练: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练状态
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_pcc = checkpoint.get('best_val_pcc', -float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"恢复训练从epoch {self.start_epoch}，最佳PCC: {self.best_val_pcc:.4f}，耐心计数: {self.patience_counter}")

    def train_epoch(self):
        """
        训练一个epoch，支持带辅助任务的模型

        返回:
        - epoch_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        aux_losses = {'contact': 0, 'intensity': 0}  # 辅助任务损失
        batch_count = 0

        # 添加变量来跟踪主任务和辅助任务的损失
        ddg_loss_sum = 0
        aux_loss_sum = 0

        # 获取总批次数
        total_batches = len(self.train_loader)

        start_time = time.time()

        for i, (wild_data, mutant_data, rna_data, ddg) in enumerate(self.train_loader):
            try:
                # 移动数据到设备
                wild_data = wild_data.to(self.device)
                mutant_data = mutant_data.to(self.device)
                rna_data = rna_data.to(self.device)

                # 处理目标数据
                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(self.device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=self.device)

                # 根据模型类型选择前向传播方式
                self.optimizer.zero_grad()

                # 检查模型是否有compute_loss方法
                if hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss')):
                    # 使用模型的compute_loss方法
                    loss, loss_info = self.model.compute_loss(wild_data, mutant_data, rna_data, target)

                    # 累计主任务和辅助任务损失
                    if isinstance(loss_info, dict):
                        if 'ddg_loss' in loss_info:
                            ddg_loss_sum += loss_info['ddg_loss'].item()
                        if 'aux_loss' in loss_info:
                            aux_loss_sum += loss_info['aux_loss'].item()

                    # 记录辅助任务损失
                    if isinstance(loss_info, dict):
                        for key, value in loss_info.items():
                            if key != 'total_loss' and key != 'ddg_loss' and key in aux_losses:
                                aux_losses[key] += value.item()
                else:
                    # 标准方式：前向传播和MSE损失
                    output = self.model(wild_data, mutant_data, rna_data)

                    # 处理输出维度
                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        output = output.unsqueeze(0)

                    # 如果返回的是元组，取第一个元素作为预测值
                    if isinstance(output, tuple):
                        output = output[0]

                    # 计算损失
                    loss = F.mse_loss(output, target)
                    ddg_loss_sum += loss.item()  # 在这种情况下，所有损失都是主任务

                # 检查NaN
                if torch.isnan(loss).any().item():
                    print(f"警告: 检测到NaN损失，跳过此批次")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 更新参数
                self.optimizer.step()

                # 累计损失
                total_loss += loss.item()
                batch_count += 1

                # 只在每个epoch的最后一个批次打印辅助任务损失详情
                if i == total_batches - 1 and hasattr(self.model, 'compute_loss'):
                    aux_weight = getattr(self.model, 'aux_weight', 0.5)  # 默认为0.5如果未指定
                    avg_ddg_loss = ddg_loss_sum / batch_count
                    avg_aux_loss = aux_loss_sum / batch_count

                    print(f"\n辅助任务损失详情 (Epoch平均):")
                    print(f"  主任务DDG损失 (MSE): {avg_ddg_loss:.6f}")
                    print(f"  总辅助损失: {avg_aux_loss:.6f}")
                    print(f"  权重因子: {aux_weight:.2f}")
                    print(f"  加权辅助损失: {(aux_weight * avg_aux_loss):.6f}")
                    print(f"  最终总损失: {(avg_ddg_loss + aux_weight * avg_aux_loss):.6f}")
                    print(
                        f"  辅助/主任务比例: {(aux_weight * avg_aux_loss / avg_ddg_loss if avg_ddg_loss > 0 else 0):.6f}")

                    # 如果有更详细的辅助任务损失，也可以打印
                    for key, value in aux_losses.items():
                        if value > 0:
                            print(f"  {key}损失: {value / batch_count:.6f}")

            except Exception as e:
                print(f"训练批次处理错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 计算平均损失
        avg_loss = total_loss / max(1, batch_count)
        epoch_time = time.time() - start_time

        # 准备辅助损失信息
        aux_info = ""
        if batch_count > 0:
            aux_losses = {k: v / batch_count for k, v in aux_losses.items() if v > 0}
            if aux_losses:
                aux_info = ", " + ", ".join([f"{k}损失: {v:.4f}" for k, v in aux_losses.items()])

        print(f"训练完成: {batch_count}批次, 平均损失: {avg_loss:.4f}{aux_info}, 用时: {epoch_time:.2f}s")

        return avg_loss

    def train_epoch_1(self):
        """
        训练一个epoch，支持带辅助任务的模型

        返回:
        - epoch_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        aux_losses = {'contact': 0, 'intensity': 0}  # 辅助任务损失
        batch_count = 0

        start_time = time.time()

        for wild_data, mutant_data, rna_data, ddg in self.train_loader:
            try:
                # 移动数据到设备
                wild_data = wild_data.to(self.device)
                mutant_data = mutant_data.to(self.device)
                rna_data = rna_data.to(self.device)

                # 处理目标数据
                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(self.device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=self.device)

                # 根据模型类型选择前向传播方式
                self.optimizer.zero_grad()

                # 检查模型是否有compute_loss方法
                if hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss')):
                    # 使用模型的compute_loss方法
                    loss, loss_info = self.model.compute_loss(wild_data, mutant_data, rna_data, target)

                    # 记录辅助任务损失
                    if isinstance(loss_info, dict):
                        for key, value in loss_info.items():
                            if key != 'total_loss' and key != 'ddg_loss' and key in aux_losses:
                                aux_losses[key] += value.item()
                else:
                    # 标准方式：前向传播和MSE损失
                    output = self.model(wild_data, mutant_data, rna_data)

                    # 处理输出维度
                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        output = output.unsqueeze(0)

                    # 如果返回的是元组，取第一个元素作为预测值
                    if isinstance(output, tuple):
                        output = output[0]

                    # 计算损失
                    loss = F.mse_loss(output, target)

                # 检查NaN
                if torch.isnan(loss).any().item():
                    print(f"警告: 检测到NaN损失，跳过此批次")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 更新参数
                self.optimizer.step()

                # 累计损失
                total_loss += loss.item()
                batch_count += 1

            except Exception as e:
                print(f"训练批次处理错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 计算平均损失
        avg_loss = total_loss / max(1, batch_count)
        epoch_time = time.time() - start_time

        # 准备辅助损失信息
        aux_info = ""
        if batch_count > 0:
            aux_losses = {k: v / batch_count for k, v in aux_losses.items() if v > 0}
            if aux_losses:
                aux_info = ", " + ", ".join([f"{k}损失: {v:.4f}" for k, v in aux_losses.items()])

        print(f"训练完成: {batch_count}批次, 平均损失: {avg_loss:.4f}{aux_info}, 用时: {epoch_time:.2f}s")

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        
        参数:
        - epoch: 当前epoch
        - is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_pcc': self.best_val_pcc,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'model_info': {
                'name': self.model.__class__.__name__,
                'module': self.model.__class__.__module__
            }
        }
        
        # 保存常规检查点
        if not is_best:
            checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # 保留最近的检查点，删除旧的
            checkpoints = sorted([
                f for f in os.listdir(os.path.join(self.output_dir, 'checkpoints'))
                if f.startswith('checkpoint_epoch_')
            ])
            
            if len(checkpoints) > 5:  # 最多保留5个检查点
                oldest_checkpoint = os.path.join(self.output_dir, 'checkpoints', checkpoints[0])
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            
            # 同时保存模型信息
            model_info = {
                'name': self.model.__class__.__name__,
                'module': self.model.__class__.__module__,
                'best_epoch': epoch,
                'best_pcc': self.best_val_pcc,
                'val_mse': self.val_metrics[-1]['mse'],
                'val_mae': self.val_metrics[-1]['mae']
            }
            
            with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"保存最佳模型 (PCC: {self.best_val_pcc:.4f}) 到 {self.best_model_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300)
        plt.close()
        
        # 绘制指标曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [m['mae'] for m in self.val_metrics], 'g-', label='MAE')
        plt.plot(epochs, [m['pcc'] for m in self.val_metrics], 'c-', label='PCC')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'metrics_curve.png'), dpi=300)
        plt.close()
    
    def train(self):
        """
        训练模型

        返回:
        - best_val_pcc: 最佳验证集PCC
        """
        print(f"开始训练模型: {self.model.__class__.__name__}")
        print(f"训练设备: {self.device}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 在验证集上评估
            val_metrics = self.evaluate()
            self.val_metrics.append(val_metrics)
            self.val_losses.append(val_metrics['loss'])

            epoch_time = time.time() - epoch_start_time

            # 更新学习率
            self.scheduler.step(val_metrics['mse'])

            # 打印进度
            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"训练损失: {train_loss:.4f} | "
                  f"验证 MSE: {val_metrics['mse']:.4f} | "
                  f"验证 MAE: {val_metrics['mae']:.4f} | "
                  f"验证 PCC: {val_metrics['pcc']:.4f} | "
                  f"耐心: {self.patience_counter}/{self.patience} | "
                  f"时间: {epoch_time:.2f}s")

            # 检查是否提高了最佳PCC
            is_best = val_metrics['pcc'] > self.best_val_pcc

            if is_best:
                self.best_val_pcc = val_metrics['pcc']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

            # 绘制训练曲线
            self.plot_training_curves()

            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"早停! {self.patience} epochs没有改善")
                break

        total_time = time.time() - start_time

        print(f"\n训练完成!")
        print(f"总训练时间: {total_time:.2f}s")
        print(f"最佳验证PCC: {self.best_val_pcc:.4f} (Epoch {self.best_epoch+1})")

        # 保存最终模型
        if self.best_epoch < self.max_epochs - 1:
            print(f"加载最佳模型 (Epoch {self.best_epoch+1})")
            best_checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

        return self.best_val_pcc

    def evaluate(self):
        """
        评估模型在验证集上的性能

        返回:
        - metrics: 评估指标字典
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        batch_count = 0

        with torch.no_grad():
            for wild_data, mutant_data, rna_data, ddg in self.val_loader:
                try:
                    # 将数据移至设备
                    wild_data = wild_data.to(self.device)
                    mutant_data = mutant_data.to(self.device)
                    rna_data = rna_data.to(self.device)

                    if isinstance(ddg, (list, tuple)):
                        ddg = ddg[0]

                    # 处理目标数据
                    if isinstance(ddg, torch.Tensor):
                        target = ddg.to(self.device)
                        if target.dim() > 1:
                            target = target.squeeze()
                    else:
                        target = torch.tensor([ddg], dtype=torch.float, device=self.device)

                    # 前向传播
                    output = self.model(wild_data, mutant_data, rna_data)

                    # 检查输出类型并提取预测值
                    if isinstance(output, dict):
                        # 如果是字典，使用'ddg'键
                        prediction = output['ddg']
                    elif isinstance(output, torch.Tensor):
                        # 如果是张量，直接使用
                        prediction = output
                    elif isinstance(output, tuple):
                        # 如果是元组，取第一个元素
                        prediction = output[0]

                    # 处理输出维度
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)

                    # 计算损失
                    loss = F.mse_loss(prediction, target)

                    # 累计损失和预测
                    total_loss += loss.item()
                    batch_count += 1
                    predictions.append(prediction.cpu())
                    targets.append(target.cpu())

                except Exception as e:
                    print(f"验证批次处理错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 计算指标
        if batch_count == 0:
            print("警告: 验证集中没有有效批次")
            return {'loss': float('inf'), 'mse': float('inf'), 'mae': float('inf'), 'pcc': float('nan')}

        # 计算平均损失和其他指标
        avg_loss = total_loss / batch_count
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()

        # 计算皮尔逊相关系数
        x = predictions - predictions.mean()
        y = targets - targets.mean()
        pcc = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2)) + 1e-8)

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'pcc': pcc.item()
        }

        return metrics

    def test(self):
        """
        在测试集上评估模型

        返回:
        - test_metrics: 测试集评估指标
        - predictions: 预测值
        - targets: 真实值
        """
        if self.test_loader is None:
            print("没有提供测试集")
            return None, None, None

        print("\n在测试集上评估模型...")

        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            # 确定测试数据的结构
            has_rna_separate = True  # 新模型始终需要RNA数据作为单独参数

            for wild_data, mutant_data, rna_data, ddg in self.test_loader:
                try:
                    # 将数据和目标移到设备上
                    wild_data = wild_data.to(self.device)
                    mutant_data = mutant_data.to(self.device)
                    rna_data = rna_data.to(self.device)

                    if isinstance(ddg, (list, tuple)):
                        ddg = ddg[0]

                    # 处理目标数据
                    if isinstance(ddg, torch.Tensor):
                        target = ddg.to(self.device)
                        if target.dim() > 1:
                            target = target.squeeze()
                    else:
                        target = torch.tensor([ddg], dtype=torch.float, device=self.device)

                    # 前向传播
                    output = self.model(wild_data, mutant_data, rna_data)

                    # 检查输出类型并提取预测值
                    if isinstance(output, dict):
                        # 如果是字典，使用'ddg'键
                        prediction = output['ddg']
                    elif isinstance(output, torch.Tensor):
                        # 如果是张量，直接使用
                        prediction = output
                    elif isinstance(output, tuple):
                        # 如果是元组，取第一个元素
                        prediction = output[0]

                    # 处理输出维度
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)

                    # 收集预测和目标
                    predictions.append(prediction.cpu())
                    targets.append(target.cpu())

                except Exception as e:
                    print(f"测试批次处理错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        if not predictions:
            print("警告: 测试集上没有有效预测")
            return None, None, None

        # 合并预测和目标
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)

        # 转换为numpy数组
        predictions_np = predictions_tensor.numpy()
        targets_np = targets_tensor.numpy()

        # 计算指标
        mse = np.mean((predictions_np - targets_np) ** 2)
        mae = np.mean(np.abs(predictions_np - targets_np))
        pcc = np.corrcoef(predictions_np.flatten(), targets_np.flatten())[0, 1]

        test_metrics = {
            'mse': mse,
            'mae': mae,
            'pcc': pcc
        }

        print(f"测试集结果:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  PCC: {pcc:.4f}")

        # 绘制预测散点图
        self.plot_predictions(predictions_np, targets_np)

        # 保存测试结果
        test_results = {
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'pcc': float(pcc)
            },
            'predictions': predictions_np.tolist(),
            'targets': targets_np.tolist()
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_metrics, predictions_np, targets_np

    # 绘制预测散点图
    def plot_predictions(self, predictions, targets):
        """
        Plot prediction scatter plot

        Parameters:
        - predictions: prediction value array
        - targets: true value array
        """
        plots_dir = os.path.join(self.output_dir, 'plots')

        # 绘制散点图和回归线
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.7)

        # 添加对角线（理想预测线）
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

        # 添加回归线
        z = np.polyfit(targets.flatten(), predictions.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(targets, p(targets), "r--", alpha=0.7)

        # 添加标签和标题
        plt.xlabel('Actual DDG Values')
        plt.ylabel('Predicted DDG Values')
        plt.title(f'Test Set Predictions (PCC: {np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]:.4f})')

        # 添加统计信息
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        pcc = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        plt.text(0.05, 0.95,
                 f'MSE: {mse:.4f}\n'
                 f'MAE: {mae:.4f}\n'
                 f'PCC: {pcc:.4f}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 保存图表
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'test_predictions.png'), dpi=300)
        plt.close()

        # 绘制直方图
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(targets, bins=20, alpha=0.7, label='Actual Values')
        plt.hist(predictions, bins=20, alpha=0.7, label='Predicted Values')
        plt.xlabel('DDG Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution Comparison: Predicted vs Actual')

        plt.subplot(1, 2, 2)
        plt.hist(predictions - targets, bins=20, alpha=0.7)
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'test_distribution.png'), dpi=300)
        plt.close()


def train_model(model, train_loader, val_loader, test_loader=None, **kwargs):
    """
    训练模型的便捷函数
    
    参数:
    - model: 模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - test_loader: 测试数据加载器
    - **kwargs: 其他参数传递给ProteinRNATrainer
    
    返回:
    - 训练器实例
    """
    trainer = ProteinRNATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        **kwargs
    )
    
    # 训练模型
    trainer.train()
    
    # 如果有测试集，进行测试
    if test_loader is not None:
        trainer.test()
    
    return trainer


# 在trainer.py中添加
class SemiSupervisedProteinRNATrainer(ProteinRNATrainer):
    """
    半监督学习训练器
    - 继承原有训练器的所有功能
    - 增加无标注结构数据的训练支持
    """

    def __init__(self, unlabeled_loader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlabeled_loader = unlabeled_loader
        self.structure_loss_history = []

    def train_epoch_semi_supervised(self):
        """
        半监督训练epoch：有监督 + 无监督结构学习

        返回:
        - epoch_loss: 平均总损失
        """
        self.model.train()
        total_supervised_loss = 0
        total_structure_loss = 0
        supervised_batches = 0
        structure_batches = 0

        start_time = time.time()

        # 1. 有监督训练（原有逻辑）
        print("进行有监督训练...")
        supervised_loss = self.train_epoch_supervised()

        # 2. 无监督结构学习
        if self.unlabeled_loader is not None and hasattr(self.model, 'compute_structure_only_loss'):
            print("进行无监督结构学习...")

            for wild_data_unlabeled in self.unlabeled_loader:
                try:
                    wild_data_unlabeled = wild_data_unlabeled.to(self.device)

                    self.optimizer.zero_grad()

                    # 计算结构预测损失
                    structure_loss = self.model.compute_structure_only_loss(wild_data_unlabeled)

                    # 应用权重
                    weighted_structure_loss = self.model.structure_loss_weight * structure_loss

                    if not torch.isnan(weighted_structure_loss):
                        weighted_structure_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                        total_structure_loss += weighted_structure_loss.item()
                        structure_batches += 1

                except Exception as e:
                    print(f"无监督批次处理错误: {str(e)}")
                    continue

        # 计算平均损失
        avg_structure_loss = total_structure_loss / max(1, structure_batches)
        self.structure_loss_history.append(avg_structure_loss)

        epoch_time = time.time() - start_time

        print(f"半监督训练完成:")
        print(f"  有监督损失: {supervised_loss:.4f}")
        print(f"  结构学习损失: {avg_structure_loss:.4f}")
        print(f"  总用时: {epoch_time:.2f}s")

        return supervised_loss  # 主要还是以有监督损失为准

    def train_epoch_supervised(self):
        """原有的有监督训练逻辑，从train_epoch中提取"""
        return super().train_epoch()

    def train(self):
        """重写训练方法，使用半监督学习"""
        print(f"开始半监督训练模型: {self.model.__class__.__name__}")

        # 替换训练方法
        original_train_epoch = self.train_epoch
        self.train_epoch = self.train_epoch_semi_supervised

        # 调用父类训练
        result = super().train()

        # 恢复原方法
        self.train_epoch = original_train_epoch

        return result

    def plot_training_curves(self):
        """扩展绘图功能，添加结构损失曲线"""
        super().plot_training_curves()

        # 绘制结构损失曲线
        if self.structure_loss_history:
            plots_dir = os.path.join(self.output_dir, 'plots')

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.structure_loss_history) + 1)
            plt.plot(epochs, self.structure_loss_history, 'g-', label='Structure Loss')
            plt.title('Unsupervised Structure Learning Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Structure Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'structure_loss_curve.png'), dpi=300)
            plt.close()