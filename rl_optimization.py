import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Any, Optional
import cv2
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """强化学习动作类型"""
    ADJUST_THRESHOLD = 0
    REFINE_POSITION = 1
    MERGE_DETECTIONS = 2
    FILTER_NOISE = 3


@dataclass
class DetectionState:
    """检测状态的数据结构"""
    image: np.ndarray
    line_heatmap: np.ndarray
    endpoint_heatmap: np.ndarray
    current_detections: Dict[str, List]
    detection_confidence: float
    processing_params: Dict[str, float]


class GeometryOptimizationEnv(gym.Env):
    """几何检测优化的强化学习环境"""
    
    def __init__(self, cnn_model, target_accuracy: float = 0.95):
        super(GeometryOptimizationEnv, self).__init__()
        
        self.cnn_model = cnn_model
        self.target_accuracy = target_accuracy
        
        # 定义动作空间：连续动作用于调整参数
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(8,),  # 8个可调参数
            dtype=np.float32
        )
        
        # 定义观察空间：图像特征 + 检测状态
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(128,),  # 压缩的特征表示
            dtype=np.float32
        )
        
        # 初始化参数范围
        self.param_ranges = {
            'line_threshold': (0.1, 0.9),
            'endpoint_threshold': (0.1, 0.9),
            'nms_threshold': (0.1, 0.8),
            'min_line_length': (10, 100),
            'max_line_gap': (1, 20),
            'corner_quality': (0.01, 0.1),
            'corner_min_distance': (5, 30),
            'gaussian_sigma': (0.5, 3.0)
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机选择一个图像进行优化
        self.current_image = self._generate_random_geometry_image()
        
        # 初始化处理参数
        self.processing_params = {
            'line_threshold': 0.5,
            'endpoint_threshold': 0.5,
            'nms_threshold': 0.4,
            'min_line_length': 30,
            'max_line_gap': 10,
            'corner_quality': 0.01,
            'corner_min_distance': 10,
            'gaussian_sigma': 1.0
        }
        
        # 获取初始检测结果
        self.current_detections = self._get_detections()
        self.step_count = 0
        self.max_steps = 20
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        
        # 根据动作调整参数
        self._apply_action(action)
        
        # 重新检测
        new_detections = self._get_detections()
        
        # 计算奖励
        reward = self._calculate_reward(new_detections)
        
        # 更新当前检测
        self.current_detections = new_detections
        
        # 检查是否结束
        done = (self.step_count >= self.max_steps) or (reward > self.target_accuracy)
        
        info = {
            'accuracy': reward,
            'detections': new_detections,
            'params': self.processing_params.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用强化学习动作到处理参数"""
        param_names = list(self.param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            if i < len(action):
                # 将动作(-1, 1)映射到参数范围
                min_val, max_val = self.param_ranges[param_name]
                delta = (action[i] * 0.1) * (max_val - min_val)
                
                new_value = self.processing_params[param_name] + delta
                new_value = np.clip(new_value, min_val, max_val)
                self.processing_params[param_name] = new_value
    
    def _get_detections(self) -> Dict[str, List]:
        """获取当前检测结果"""
        # 使用CNN模型获取热力图
        with torch.no_grad():
            image_tensor = torch.FloatTensor(self.current_image).unsqueeze(0).unsqueeze(0)
            outputs = self.cnn_model(image_tensor)
            
            line_heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
            endpoint_heatmap = outputs['endpoint_heatmap'][0, 0].cpu().numpy()
        
        # 后处理获取检测结果
        lines = self._extract_lines_from_heatmap(line_heatmap)
        endpoints = self._extract_endpoints_from_heatmap(endpoint_heatmap)
        
        return {
            'lines': lines,
            'endpoints': endpoints,
            'line_heatmap': line_heatmap,
            'endpoint_heatmap': endpoint_heatmap
        }
    
    def _extract_lines_from_heatmap(self, heatmap: np.ndarray) -> List[List[int]]:
        """从热力图中提取线条"""
        # 阈值化
        binary = (heatmap > self.processing_params['line_threshold']).astype(np.uint8) * 255
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=int(self.processing_params['min_line_length']),
            minLineLength=int(self.processing_params['min_line_length']),
            maxLineGap=int(self.processing_params['max_line_gap'])
        )
        
        if lines is not None:
            return lines.reshape(-1, 4).tolist()
        return []
    
    def _extract_endpoints_from_heatmap(self, heatmap: np.ndarray) -> List[Tuple[int, int]]:
        """从热力图中提取端点"""
        # 阈值化
        binary = (heatmap > self.processing_params['endpoint_threshold']).astype(np.uint8) * 255
        
        # 高斯滤波
        binary = cv2.GaussianBlur(binary, (5, 5), self.processing_params['gaussian_sigma'])
        
        # 角点检测
        corners = cv2.goodFeaturesToTrack(
            binary,
            maxCorners=100,
            qualityLevel=self.processing_params['corner_quality'],
            minDistance=self.processing_params['corner_min_distance']
        )
        
        endpoints = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                endpoints.append((int(x), int(y)))
        
        return endpoints
    
    def _calculate_reward(self, detections: Dict[str, List]) -> float:
        """计算奖励函数"""
        # 这里应该与真实标注比较，为了演示我们使用一些启发式方法
        lines = detections['lines']
        endpoints = detections['endpoints']
        
        # 基础分数
        line_score = min(len(lines) / 5.0, 1.0)  # 假设理想情况下有5条线
        endpoint_score = min(len(endpoints) / 10.0, 1.0)  # 假设理想情况下有10个端点
        
        # 几何一致性检查
        consistency_score = self._check_geometric_consistency(lines, endpoints)
        
        # 检测质量评分
        quality_score = self._evaluate_detection_quality(detections)
        
        # 综合奖励
        total_reward = 0.3 * line_score + 0.3 * endpoint_score + 0.2 * consistency_score + 0.2 * quality_score
        
        return total_reward
    
    def _check_geometric_consistency(self, lines: List[List[int]], endpoints: List[Tuple[int, int]]) -> float:
        """检查几何一致性"""
        if not lines or not endpoints:
            return 0.0
        
        # 检查端点是否在线条附近
        consistent_endpoints = 0
        for endpoint in endpoints:
            for line in lines:
                x1, y1, x2, y2 = line
                # 计算点到线的距离
                dist = self._point_to_line_distance(endpoint, (x1, y1), (x2, y2))
                if dist < 10:  # 阈值
                    consistent_endpoints += 1
                    break
        
        consistency = consistent_endpoints / len(endpoints) if endpoints else 0
        return consistency
    
    def _point_to_line_distance(self, point: Tuple[int, int], 
                               line_start: Tuple[int, int], 
                               line_end: Tuple[int, int]) -> float:
        """计算点到线的距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 使用公式计算点到线段的距离
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx, yy = x1 + param * C, y1 + param * D
        
        dx = x0 - xx
        dy = y0 - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def _evaluate_detection_quality(self, detections: Dict[str, List]) -> float:
        """评估检测质量"""
        heatmap_line = detections['line_heatmap']
        heatmap_endpoint = detections['endpoint_heatmap']
        
        # 计算热力图的信噪比
        line_snr = np.std(heatmap_line) / (np.mean(heatmap_line) + 1e-8)
        endpoint_snr = np.std(heatmap_endpoint) / (np.mean(heatmap_endpoint) + 1e-8)
        
        quality = (line_snr + endpoint_snr) / 2.0
        return np.clip(quality, 0, 1)
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        # 提取图像特征
        image_features = self._extract_image_features()
        
        # 提取检测状态特征
        detection_features = self._extract_detection_features()
        
        # 提取参数状态
        param_features = self._extract_param_features()
        
        # 合并所有特征
        observation = np.concatenate([image_features, detection_features, param_features])
        return observation.astype(np.float32)
    
    def _extract_image_features(self) -> np.ndarray:
        """提取图像特征"""
        # 简单的统计特征
        img = self.current_image
        features = [
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            # 添加更多特征...
        ]
        # 填充到固定长度
        features.extend([0.0] * (64 - len(features)))
        return np.array(features[:64])
    
    def _extract_detection_features(self) -> np.ndarray:
        """提取检测特征"""
        lines = self.current_detections.get('lines', [])
        endpoints = self.current_detections.get('endpoints', [])
        
        features = [
            len(lines),
            len(endpoints),
            np.mean([self._line_length(line) for line in lines]) if lines else 0,
            # 添加更多特征...
        ]
        # 填充到固定长度
        features.extend([0.0] * (32 - len(features)))
        return np.array(features[:32])
    
    def _extract_param_features(self) -> np.ndarray:
        """提取参数特征"""
        params = list(self.processing_params.values())
        # 标准化参数值
        normalized_params = []
        for i, (param_name, value) in enumerate(self.processing_params.items()):
            min_val, max_val = self.param_ranges[param_name]
            normalized = (value - min_val) / (max_val - min_val)
            normalized_params.append(normalized)
        
        # 填充到固定长度
        while len(normalized_params) < 32:
            normalized_params.append(0.0)
        
        return np.array(normalized_params[:32])
    
    def _line_length(self, line: List[int]) -> float:
        """计算线条长度"""
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _generate_random_geometry_image(self) -> np.ndarray:
        """生成随机几何图像用于训练"""
        img = np.zeros((512, 512), dtype=np.uint8)
        
        # 随机绘制一些线条
        num_lines = np.random.randint(3, 8)
        for _ in range(num_lines):
            pt1 = (np.random.randint(50, 462), np.random.randint(50, 462))
            pt2 = (np.random.randint(50, 462), np.random.randint(50, 462))
            cv2.line(img, pt1, pt2, 255, 2)
        
        # 添加噪声
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img


class GeometryRLOptimizer:
    """几何检测的强化学习优化器"""
    
    def __init__(self, cnn_model, algorithm: str = "PPO"):
        self.cnn_model = cnn_model
        self.algorithm = algorithm
        self.env = None
        self.model = None
        
        self._setup_environment()
        self._setup_rl_model()
    
    def _setup_environment(self):
        """设置强化学习环境"""
        self.env = GeometryOptimizationEnv(self.cnn_model)
    
    def _setup_rl_model(self):
        """设置强化学习模型"""
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, total_timesteps: int = 50000):
        """训练强化学习模型"""
        print(f"开始训练 {self.algorithm} 模型...")
        
        # 自定义回调函数
        callback = TrainingCallback()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        print("训练完成！")
    
    def optimize_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """使用训练好的模型优化检测结果"""
        # 设置环境的图像
        self.env.current_image = image
        obs = self.env.reset()
        
        # 使用训练好的模型进行优化
        done = False
        step_count = 0
        max_steps = 20
        
        while not done and step_count < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            step_count += 1
        
        return {
            'optimized_detections': info['detections'],
            'final_params': info['params'],
            'final_accuracy': info['accuracy'],
            'optimization_steps': step_count
        }
    
    def save_model(self, path: str):
        """保存训练好的模型"""
        self.model.save(path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载训练好的模型"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=self.env)
        print(f"模型已从 {path} 加载")


class TrainingCallback(BaseCallback):
    """训练过程中的回调函数"""
    
    def __init__(self):
        super(TrainingCallback, self).__init__()
        self.best_reward = -np.inf
    
    def _on_step(self) -> bool:
        # 记录训练进度
        if len(self.locals.get('rewards', [])) > 0:
            current_reward = np.mean(self.locals['rewards'])
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                print(f"新的最佳奖励: {self.best_reward:.4f}")
        
        return True


def create_rl_optimizer(cnn_model, algorithm: str = "PPO") -> GeometryRLOptimizer:
    """创建强化学习优化器的工厂函数"""
    return GeometryRLOptimizer(cnn_model, algorithm)


if __name__ == "__main__":
    # 这里需要加载训练好的CNN模型
    from cnn_models import create_model
    
    # 创建模型（这里使用随机初始化的模型作为示例）
    cnn_model = create_model("resnet")
    cnn_model.eval()
    
    # 创建强化学习优化器
    rl_optimizer = create_rl_optimizer(cnn_model, "PPO")
    
    # 训练模型
    print("开始强化学习训练...")
    rl_optimizer.train(total_timesteps=10000)  # 减少时间步数用于快速测试
    
    print("强化学习优化模块创建完成！")