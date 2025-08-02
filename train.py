#!/usr/bin/env python3
"""
几何检测模型训练脚本

这个脚本展示了如何训练CNN模型和强化学习优化器
"""

import os
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 导入自定义模块
from data_preprocessing import GeometryImageProcessor, GeometryDataset
from cnn_models import create_model, GeometryLoss
from rl_optimization import create_rl_optimizer
from visualization import create_visualizer


class CNNTrainer:
    """CNN模型训练器"""
    
    def __init__(self, 
                 model_type: str = "resnet",
                 device: str = "auto",
                 learning_rate: float = 1e-3,
                 batch_size: int = 8):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型
            device: 计算设备
            learning_rate: 学习率
            batch_size: 批量大小
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = create_model(model_type).to(self.device)
        self.model_type = model_type
        
        # 创建损失函数和优化器
        self.criterion = GeometryLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # 训练参数
        self.batch_size = batch_size
        
        # TensorBoard
        self.writer = SummaryWriter('runs/geometry_detection')
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, num_samples: int = 1000, val_split: float = 0.2):
        """
        准备训练数据
        
        Args:
            num_samples: 样本数量
            val_split: 验证集比例
        """
        print("生成训练数据...")
        
        # 创建数据处理器
        processor = GeometryImageProcessor()
        
        # 生成合成数据
        images, annotations = processor.generate_synthetic_geometry(num_samples)
        
        # 创建训练和验证数据集
        val_size = int(len(images) * val_split)
        train_size = len(images) - val_size
        
        train_images = images[:train_size]
        train_annotations = annotations[:train_size]
        val_images = images[train_size:]
        val_annotations = annotations[train_size:]
        
        # 创建数据集和数据加载器
        self.train_dataset = GeometryDataset(train_images, train_annotations, augment=True)
        self.val_dataset = GeometryDataset(val_images, val_annotations, augment=False)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        print(f"训练样本: {len(self.train_dataset)}")
        print(f"验证样本: {len(self.val_dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_line_loss = 0.0
        running_endpoint_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            images = batch['image'].to(self.device)
            annotations = batch['annotation']
            
            # 创建目标热力图
            targets = self._create_targets(annotations, images.shape[-2:])
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            losses = self.criterion(outputs, targets)
            total_loss = losses['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 累积损失
            running_loss += total_loss.item()
            running_line_loss += losses.get('line_loss', 0).item()
            running_endpoint_loss += losses.get('endpoint_loss', 0).item()
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {total_loss.item():.4f}')
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.train_loader)
        epoch_line_loss = running_line_loss / len(self.train_loader)
        epoch_endpoint_loss = running_endpoint_loss / len(self.train_loader)
        
        return {
            'total_loss': epoch_loss,
            'line_loss': epoch_line_loss,
            'endpoint_loss': epoch_endpoint_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_line_loss = 0.0
        running_endpoint_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 获取数据
                images = batch['image'].to(self.device)
                annotations = batch['annotation']
                
                # 创建目标热力图
                targets = self._create_targets(annotations, images.shape[-2:])
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                losses = self.criterion(outputs, targets)
                
                # 累积损失
                running_loss += losses['total_loss'].item()
                running_line_loss += losses.get('line_loss', 0).item()
                running_endpoint_loss += losses.get('endpoint_loss', 0).item()
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.val_loader)
        epoch_line_loss = running_line_loss / len(self.val_loader)
        epoch_endpoint_loss = running_endpoint_loss / len(self.val_loader)
        
        return {
            'total_loss': epoch_loss,
            'line_loss': epoch_line_loss,
            'endpoint_loss': epoch_endpoint_loss
        }
    
    def _create_targets(self, annotations: List[Dict], image_shape: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """创建训练目标"""
        batch_size = len(annotations)
        height, width = image_shape
        
        # 初始化目标张量
        line_targets = torch.zeros(batch_size, 1, height, width).to(self.device)
        endpoint_targets = torch.zeros(batch_size, 1, height, width).to(self.device)
        
        processor = GeometryImageProcessor()
        
        for i, annotation in enumerate(annotations):
            if 'lines' in annotation and 'endpoints' in annotation:
                # 创建热力图
                line_heatmap, endpoint_heatmap = processor.create_heatmaps(
                    (height, width), 
                    annotation['endpoints'], 
                    annotation['lines']
                )
                
                line_targets[i, 0] = torch.from_numpy(line_heatmap)
                endpoint_targets[i, 0] = torch.from_numpy(endpoint_heatmap)
        
        return {
            'line_heatmap': line_targets,
            'endpoint_heatmap': endpoint_targets
        }
    
    def train(self, num_epochs: int = 50, save_path: str = "models/cnn_model.pth"):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_path: 模型保存路径
        """
        print(f"开始训练，总轮数: {num_epochs}")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录损失
            train_losses.append(train_metrics['total_loss'])
            val_losses.append(val_metrics['total_loss'])
            
            # TensorBoard日志
            self.writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Train_Line', train_metrics['line_loss'], epoch)
            self.writer.add_scalar('Loss/Train_Endpoint', train_metrics['endpoint_loss'], epoch)
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {train_metrics["total_loss"]:.4f}')
            print(f'  验证损失: {val_metrics["total_loss"]:.4f}')
            print(f'  耗时: {epoch_time:.2f}秒')
            print('-' * 50)
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                    'model_type': self.model_type
                }, save_path)
                print(f'保存最佳模型到: {save_path}')
        
        self.writer.close()
        
        # 绘制训练曲线
        self._plot_training_curves(train_losses, val_losses)
        
        print("训练完成！")
    
    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', color='blue')
        plt.plot(val_losses, label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("训练曲线已保存到: training_curves.png")


def train_cnn_model(args):
    """训练CNN模型"""
    print("=" * 60)
    print("开始训练CNN模型")
    print("=" * 60)
    
    # 创建训练器
    trainer = CNNTrainer(
        model_type=args.model,
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    # 准备数据
    trainer.prepare_data(num_samples=args.num_samples)
    
    # 训练模型
    trainer.train(num_epochs=args.epochs, save_path=args.save_path)


def train_rl_optimizer(args):
    """训练强化学习优化器"""
    print("=" * 60)
    print("开始训练强化学习优化器")
    print("=" * 60)
    
    # 创建预训练的CNN模型
    cnn_model = create_model(args.model)
    
    # 加载预训练权重（如果存在）
    if os.path.exists(args.cnn_model_path):
        checkpoint = torch.load(args.cnn_model_path)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载预训练CNN模型: {args.cnn_model_path}")
    else:
        print("警告: 未找到预训练CNN模型，使用随机初始化")
    
    cnn_model.eval()
    
    # 创建强化学习优化器
    rl_optimizer = create_rl_optimizer(cnn_model, algorithm=args.rl_algorithm)
    
    # 训练
    rl_optimizer.train(total_timesteps=args.rl_steps)
    
    # 保存模型
    os.makedirs(os.path.dirname(args.rl_save_path), exist_ok=True)
    rl_optimizer.save_model(args.rl_save_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="几何检测模型训练")
    
    # 通用参数
    parser.add_argument("--mode", type=str, choices=["cnn", "rl", "both"], default="both",
                       help="训练模式")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "unet"],
                       help="CNN模型类型")
    
    # CNN训练参数
    parser.add_argument("--epochs", type=int, default=50, help="CNN训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-samples", type=int, default=1000, help="训练样本数量")
    parser.add_argument("--save-path", type=str, default="models/cnn_model.pth",
                       help="CNN模型保存路径")
    
    # 强化学习训练参数
    parser.add_argument("--rl-algorithm", type=str, default="PPO", choices=["PPO", "SAC"],
                       help="强化学习算法")
    parser.add_argument("--rl-steps", type=int, default=50000, help="强化学习训练步数")
    parser.add_argument("--cnn-model-path", type=str, default="models/cnn_model.pth",
                       help="预训练CNN模型路径")
    parser.add_argument("--rl-save-path", type=str, default="models/rl_optimizer.zip",
                       help="强化学习模型保存路径")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "cnn":
            train_cnn_model(args)
        elif args.mode == "rl":
            train_rl_optimizer(args)
        elif args.mode == "both":
            # 先训练CNN，再训练强化学习
            train_cnn_model(args)
            print("\nCNN训练完成，开始训练强化学习优化器...\n")
            train_rl_optimizer(args)
        
        print("\n" + "=" * 60)
        print("所有训练完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()