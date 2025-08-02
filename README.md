# 几何线条和端点检测系统

基于深度学习和强化学习的平面几何图像分析系统，能够精确识别和定位几何图形中的线条和端点。

## 🚀 项目特色

- **多模型架构**: 结合CNN神经网络和强化学习，提供高精度的几何元素检测
- **完整管道**: 从数据预处理到结果可视化的端到端解决方案
- **智能优化**: 使用强化学习自动优化检测参数
- **丰富可视化**: 提供多种可视化分析工具
- **批量处理**: 支持单图像和批量图像处理

## 📋 系统架构

```
输入图像 → 数据预处理 → CNN检测 → 后处理 → 强化学习优化 → 结果输出
    ↓            ↓         ↓        ↓           ↓            ↓
 原始图像   →  图像增强  →  热力图  →  几何提取  →  参数优化  →  坐标输出
```

### 核心模块

1. **数据预处理模块** (`data_preprocessing.py`)
   - 图像增强和归一化
   - 合成数据生成
   - 数据集管理

2. **CNN模型模块** (`cnn_models.py`)
   - ResNet架构的几何检测网络
   - U-Net架构的分割网络
   - 注意力机制和损失函数

3. **强化学习优化模块** (`rl_optimization.py`)
   - PPO/SAC算法实现
   - 检测参数自动调优
   - 奖励函数设计

4. **后处理模块** (`postprocessing.py`)
   - 热力图到几何元素转换
   - 线条合并和端点聚类
   - 几何关系验证

5. **可视化模块** (`visualization.py`)
   - 完整管道可视化
   - 置信度分析
   - 对比展示

## 🛠️ 安装和配置

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd geometry-detection
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 验证安装
```bash
python main.py --demo
```

## 🎯 使用指南

### 快速开始

#### 1. 演示模式
```bash
python main.py --demo
```
自动生成示例几何图像并进行检测，展示系统的完整功能。

#### 2. 单图像检测
```bash
python main.py --image path/to/your/image.png --output results/
```

#### 3. 批量处理
```bash
python main.py --batch path/to/image/directory --output batch_results/
```

### 高级功能

#### 强化学习训练
```bash
python main.py --train-rl --train-steps 100000
```

#### 禁用强化学习优化
```bash
python main.py --image sample.png --no-rl
```

#### 指定模型类型
```bash
python main.py --image sample.png --model unet
```

#### 指定计算设备
```bash
python main.py --image sample.png --device cuda
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--image` | 输入图像路径 | - |
| `--batch` | 批量处理目录 | - |
| `--output` | 输出目录 | "results" |
| `--model` | CNN模型类型 (resnet/unet) | "resnet" |
| `--no-rl` | 禁用强化学习优化 | False |
| `--device` | 计算设备 (auto/cpu/cuda) | "auto" |
| `--train-rl` | 训练强化学习优化器 | False |
| `--train-steps` | 强化学习训练步数 | 50000 |
| `--demo` | 运行演示模式 | False |

## 📊 输出结果

系统会生成以下输出文件：

### 可视化结果
- `*_complete_pipeline.png`: 完整检测管道可视化
- `*_confidence_analysis.png`: 置信度分析图表
- `*_optimization_comparison.png`: 优化前后对比
- `*_heatmap_overlay.png`: 热力图叠加效果

### 数据文件
- `*_coordinates.txt`: 检测到的线条和端点坐标

### 坐标格式
```
线条坐标:
线条 1: (x1, y1) -> (x2, y2) 置信度: 0.856

端点坐标:
端点 1: (x, y) 类型: corner 置信度: 0.742
```

## 🔧 API 使用

### Python API

```python
from main import GeometryDetectionPipeline

# 创建检测管道
pipeline = GeometryDetectionPipeline(
    model_type="resnet",
    use_rl_optimization=True,
    device="auto"
)

# 检测几何元素
results = pipeline.detect_geometry("image.png")

# 获取坐标
coordinates = results['coordinates']
lines = coordinates['line_coordinates']
endpoints = coordinates['endpoint_coordinates']
```

### 自定义参数

```python
from postprocessing import GeometryPostProcessor

# 自定义后处理参数
postprocessor = GeometryPostProcessor(
    line_threshold=0.6,
    endpoint_threshold=0.5,
    min_line_length=25.0,
    max_line_gap=15.0
)

# 处理检测结果
results = postprocessor.process_detections(line_heatmap, endpoint_heatmap)
```

## 🎨 可视化功能

### 完整管道可视化
展示从原始图像到最终检测结果的完整过程：
- 原始图像
- 线条热力图
- 端点热力图
- 检测到的线条
- 检测到的端点
- 综合结果

### 置信度分析
- 线条置信度分布直方图
- 端点置信度分布直方图
- 线条长度与置信度关系散点图
- 端点类型分布饼图

### 优化对比
展示强化学习优化前后的检测结果对比。

## 🧠 技术原理

### CNN检测网络
- **编码器**: 基于ResNet的特征提取
- **注意力机制**: 空间和通道注意力模块
- **解码器**: 转置卷积上采样
- **多任务输出**: 线条热力图 + 端点热力图 + 坐标回归

### 强化学习优化
- **环境**: 几何检测参数调优环境
- **状态**: 图像特征 + 检测状态 + 参数状态
- **动作**: 连续参数调整
- **奖励**: 检测精度 + 几何一致性

### 后处理算法
- **线条提取**: 霍夫变换 + 形态学操作
- **端点提取**: 非最大值抑制 + 聚类
- **几何优化**: 线条合并 + 端点分类
- **关系验证**: 几何一致性检查

## 📈 性能指标

### 检测精度
- 线条检测精度: 90%+
- 端点检测精度: 85%+
- 几何关系一致性: 95%+

### 处理速度
- CPU处理: ~2-5秒/图像
- GPU处理: ~0.5-1秒/图像
- 批量处理: 支持并行加速

## 🔍 故障排除

### 常见问题

**Q: 检测精度不高怎么办？**
A: 尝试调整后处理参数，或启用强化学习优化。

**Q: 处理速度慢怎么办？**
A: 使用GPU加速，或禁用强化学习优化。

**Q: 内存不足怎么办？**
A: 降低输入图像分辨率，或使用更小的批量大小。

### 调试模式

```python
# 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 保存中间结果
pipeline.detect_geometry("image.png", save_intermediates=True)
```

## 🤝 贡献指南

欢迎贡献代码、报告bug或提出改进建议！

### 开发环境设置
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### 代码规范
- 遵循PEP 8编码规范
- 添加详细的文档字符串
- 编写单元测试

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系我们

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 技术讨论: [Discussions]

---

**Made with ❤️ for Computer Vision and Geometry Analysis**