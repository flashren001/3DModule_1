import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, List
import numpy as np


class ResidualBlock(nn.Module):
    """残差块，用于构建更深的网络"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x


class GeometryDetectionNet(nn.Module):
    """几何检测网络，用于同时检测线条和端点"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 2):
        super(GeometryDetectionNet, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 残差块
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
        
        # 注意力机制
        self.attention = AttentionModule(512)
        
        # 解码器部分 - 用于语义分割
        self.decoder = nn.Sequential(
            # 上采样层
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # 输出层 - 线条和端点的热力图
        self.line_head = nn.Conv2d(16, 1, kernel_size=1)
        self.endpoint_head = nn.Conv2d(16, 1, kernel_size=1)
        
        # 坐标回归头
        self.coord_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  # 最多16个端点的坐标 (16 * 2 * 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码
        features = self.encoder(x)
        
        # 注意力
        attended_features = self.attention(features)
        
        # 解码
        decoded = self.decoder(attended_features)
        
        # 生成热力图
        line_heatmap = torch.sigmoid(self.line_head(decoded))
        endpoint_heatmap = torch.sigmoid(self.endpoint_head(decoded))
        
        # 坐标回归
        coords = self.coord_regressor(attended_features)
        
        return {
            'line_heatmap': line_heatmap,
            'endpoint_heatmap': endpoint_heatmap,
            'coordinates': coords
        }


class UNetGeometryDetector(nn.Module):
    """基于U-Net的几何检测网络"""
    
    def __init__(self, input_channels: int = 1):
        super(UNetGeometryDetector, self).__init__()
        
        # 编码器
        self.enc1 = self._double_conv(input_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        self.enc5 = self._double_conv(512, 1024)
        
        # 池化层
        self.pool = nn.MaxPool2d(2)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # 输出层
        self.line_output = nn.Conv2d(64, 1, kernel_size=1)
        self.endpoint_output = nn.Conv2d(64, 1, kernel_size=1)
    
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """双重卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # 解码路径
        d4 = self.dec4(torch.cat([self.upconv4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        # 输出
        line_heatmap = torch.sigmoid(self.line_output(d1))
        endpoint_heatmap = torch.sigmoid(self.endpoint_output(d1))
        
        return {
            'line_heatmap': line_heatmap,
            'endpoint_heatmap': endpoint_heatmap
        }


class GeometryLoss(nn.Module):
    """几何检测的损失函数"""
    
    def __init__(self, line_weight: float = 1.0, endpoint_weight: float = 2.0, 
                 coord_weight: float = 0.5):
        super(GeometryLoss, self).__init__()
        self.line_weight = line_weight
        self.endpoint_weight = endpoint_weight
        self.coord_weight = coord_weight
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0
        
        # 线条热力图损失
        if 'line_heatmap' in predictions and 'line_heatmap' in targets:
            line_loss = self.focal_loss(predictions['line_heatmap'], targets['line_heatmap'])
            losses['line_loss'] = line_loss
            total_loss += self.line_weight * line_loss
        
        # 端点热力图损失
        if 'endpoint_heatmap' in predictions and 'endpoint_heatmap' in targets:
            endpoint_loss = self.focal_loss(predictions['endpoint_heatmap'], targets['endpoint_heatmap'])
            losses['endpoint_loss'] = endpoint_loss
            total_loss += self.endpoint_weight * endpoint_loss
        
        # 坐标回归损失
        if 'coordinates' in predictions and 'coordinates' in targets:
            coord_loss = self.mse_loss(predictions['coordinates'], targets['coordinates'])
            losses['coord_loss'] = coord_loss
            total_loss += self.coord_weight * coord_loss
        
        losses['total_loss'] = total_loss
        return losses


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def create_model(model_type: str = "resnet", input_channels: int = 1) -> nn.Module:
    """创建指定类型的模型"""
    if model_type == "resnet":
        return GeometryDetectionNet(input_channels)
    elif model_type == "unet":
        return UNetGeometryDetector(input_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_pretrained_backbone(model: nn.Module, backbone_type: str = "resnet50") -> nn.Module:
    """加载预训练的骨干网络"""
    if backbone_type == "resnet50":
        pretrained = models.resnet50(pretrained=True)
        # 这里可以添加将预训练权重转移到模型的逻辑
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_model("resnet").to(device)
    
    # 测试输入
    test_input = torch.randn(2, 1, 512, 512).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_input)
    
    print("模型输出形状:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print("CNN模型创建完成！")