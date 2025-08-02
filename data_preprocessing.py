import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import feature, morphology
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Tuple, List, Dict


class GeometryDataset(Dataset):
    """几何图形数据集类，用于加载和预处理几何图像"""
    
    def __init__(self, images: List[np.ndarray], annotations: List[Dict] = None, 
                 transform=None, augment=True):
        self.images = images
        self.annotations = annotations or []
        self.transform = transform
        self.augment = augment
        
        # 定义数据增强管道
        if augment:
            self.augmentation = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)
                ], p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        else:
            self.augmentation = A.Compose([
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # 确保图像是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用增强
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        sample = {'image': image}
        
        if idx < len(self.annotations):
            sample['annotation'] = self.annotations[idx]
            
        return sample


class GeometryImageProcessor:
    """几何图像处理器，提供图像预处理和特征提取功能"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理输入图像"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 调整大小
        resized = cv2.resize(gray, self.target_size)
        
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # 直方图均衡化
        equalized = cv2.equalizeHist(denoised)
        
        return equalized
    
    def extract_lines_hough(self, image: np.ndarray) -> List[np.ndarray]:
        """使用霍夫变换提取直线"""
        # 边缘检测
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=30,
            maxLineGap=10
        )
        
        return lines if lines is not None else []
    
    def extract_endpoints(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """提取线条端点"""
        # 边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        # 角点检测
        corners = cv2.goodFeaturesToTrack(
            edges,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3,
            useHarrisDetector=True,
            k=0.04
        )
        
        endpoints = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                endpoints.append((int(x), int(y)))
        
        return endpoints
    
    def generate_synthetic_geometry(self, num_samples: int = 1000) -> Tuple[List[np.ndarray], List[Dict]]:
        """生成合成几何图形数据"""
        images = []
        annotations = []
        
        for _ in range(num_samples):
            # 创建空白图像
            img = np.zeros(self.target_size, dtype=np.uint8)
            annotation = {'lines': [], 'endpoints': []}
            
            # 随机生成线条数量
            num_lines = random.randint(2, 6)
            
            for _ in range(num_lines):
                # 随机生成线条端点
                pt1 = (random.randint(50, self.target_size[0]-50), 
                       random.randint(50, self.target_size[1]-50))
                pt2 = (random.randint(50, self.target_size[0]-50), 
                       random.randint(50, self.target_size[1]-50))
                
                # 绘制线条
                cv2.line(img, pt1, pt2, 255, 2)
                
                # 记录标注
                annotation['lines'].append([pt1[0], pt1[1], pt2[0], pt2[1]])
                annotation['endpoints'].extend([pt1, pt2])
            
            # 去除重复端点
            annotation['endpoints'] = list(set(annotation['endpoints']))
            
            # 添加噪声
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            images.append(img)
            annotations.append(annotation)
        
        return images, annotations
    
    def create_heatmaps(self, image_shape: Tuple[int, int], 
                       endpoints: List[Tuple[int, int]], 
                       lines: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """创建端点和线条的热力图"""
        endpoint_heatmap = np.zeros(image_shape, dtype=np.float32)
        line_heatmap = np.zeros(image_shape, dtype=np.float32)
        
        # 端点热力图
        for x, y in endpoints:
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(endpoint_heatmap, (x, y), 5, 1.0, -1)
        
        # 线条热力图
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_heatmap, (x1, y1), (x2, y2), 1.0, 3)
        
        # 高斯模糊使热力图更平滑
        endpoint_heatmap = cv2.GaussianBlur(endpoint_heatmap, (11, 11), 3)
        line_heatmap = cv2.GaussianBlur(line_heatmap, (7, 7), 2)
        
        return endpoint_heatmap, line_heatmap


def visualize_detection_results(image: np.ndarray, 
                              detected_lines: List[np.ndarray], 
                              detected_endpoints: List[Tuple[int, int]],
                              title: str = "Detection Results"):
    """可视化检测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 检测结果
    result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
    
    # 绘制检测到的线条
    for line in detected_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制检测到的端点
    for x, y in detected_endpoints:
        cv2.circle(result_img, (x, y), 5, (255, 0, 0), -1)
    
    axes[1].imshow(result_img)
    axes[1].set_title(f'{title}\nLines: {len(detected_lines)}, Endpoints: {len(detected_endpoints)}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例用法
    processor = GeometryImageProcessor()
    
    # 生成合成数据
    print("生成合成几何数据...")
    images, annotations = processor.generate_synthetic_geometry(num_samples=100)
    
    # 创建数据集
    dataset = GeometryDataset(images, annotations, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"数据集大小: {len(dataset)}")
    print("数据预处理模块创建完成！")