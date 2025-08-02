#!/usr/bin/env python3
"""
几何线条和端点检测主程序
结合CNN神经网络和强化学习的平面几何图像分析系统

作者：AI助手
版本：1.0
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入自定义模块
from data_preprocessing import GeometryImageProcessor, GeometryDataset
from cnn_models import create_model, GeometryLoss
from rl_optimization import create_rl_optimizer
from postprocessing import GeometryPostProcessor, convert_to_coordinates
from visualization import create_visualizer


class GeometryDetectionPipeline:
    """几何检测完整管道"""
    
    def __init__(self, 
                 model_type: str = "resnet",
                 use_rl_optimization: bool = True,
                 device: str = "auto"):
        """
        初始化检测管道
        
        Args:
            model_type: CNN模型类型 ("resnet" 或 "unet")
            use_rl_optimization: 是否使用强化学习优化
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.model_type = model_type
        self.use_rl_optimization = use_rl_optimization
        
        # 设置设备
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("初始化检测管道组件...")
        
        # 1. 数据预处理器
        self.preprocessor = GeometryImageProcessor()
        
        # 2. CNN模型
        self.cnn_model = create_model(self.model_type).to(self.device)
        self.cnn_model.eval()
        
        # 3. 后处理器
        self.postprocessor = GeometryPostProcessor()
        
        # 4. 强化学习优化器（可选）
        if self.use_rl_optimization:
            try:
                self.rl_optimizer = create_rl_optimizer(self.cnn_model)
                print("强化学习优化器已初始化")
            except Exception as e:
                print(f"强化学习优化器初始化失败: {e}")
                self.use_rl_optimization = False
        
        # 5. 可视化器
        self.visualizer = create_visualizer()
        
        print("所有组件初始化完成！")
    
    def detect_geometry(self, 
                       image_path: str, 
                       save_results: bool = True,
                       output_dir: str = "results") -> Dict[str, Any]:
        """
        检测图像中的几何元素
        
        Args:
            image_path: 输入图像路径
            save_results: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            检测结果字典
        """
        print(f"开始处理图像: {image_path}")
        start_time = time.time()
        
        # 1. 加载和预处理图像
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        processed_image = self.preprocessor.preprocess_image(original_image)
        
        # 2. CNN推理
        print("执行CNN推理...")
        cnn_results = self._run_cnn_inference(processed_image)
        
        # 3. 初始后处理
        print("执行后处理...")
        initial_results = self.postprocessor.process_detections(
            cnn_results['line_heatmap'], 
            cnn_results['endpoint_heatmap']
        )
        
        # 4. 强化学习优化（可选）
        optimized_results = initial_results
        if self.use_rl_optimization:
            print("执行强化学习优化...")
            try:
                optimization_output = self.rl_optimizer.optimize_detection(processed_image)
                optimized_results = optimization_output['optimized_detections']
                print(f"优化完成，用了 {optimization_output['optimization_steps']} 步")
            except Exception as e:
                print(f"强化学习优化失败: {e}")
        
        # 5. 转换为坐标格式
        final_coordinates = convert_to_coordinates(optimized_results)
        
        # 6. 创建结果字典
        results = {
            'original_image': original_image,
            'processed_image': processed_image,
            'cnn_outputs': cnn_results,
            'initial_detections': initial_results,
            'optimized_detections': optimized_results,
            'coordinates': final_coordinates,
            'processing_time': time.time() - start_time,
            'image_path': image_path
        }
        
        # 7. 保存结果
        if save_results:
            self._save_results(results, output_dir)
        
        print(f"处理完成！耗时: {results['processing_time']:.2f}秒")
        print(f"检测到 {len(optimized_results['lines'])} 条线条和 {len(optimized_results['endpoints'])} 个端点")
        
        return results
    
    def _run_cnn_inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """运行CNN推理"""
        # 准备输入
        input_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.cnn_model(input_tensor)
        
        # 转换输出
        line_heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
        endpoint_heatmap = outputs['endpoint_heatmap'][0, 0].cpu().numpy()
        
        return {
            'line_heatmap': line_heatmap,
            'endpoint_heatmap': endpoint_heatmap
        }
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """保存检测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        base_name = Path(results['image_path']).stem
        
        # 1. 保存可视化结果
        self.visualizer.visualize_complete_pipeline(
            results['original_image'],
            results['cnn_outputs']['line_heatmap'],
            results['cnn_outputs']['endpoint_heatmap'],
            results['optimized_detections']['lines'],
            results['optimized_detections']['endpoints'],
            save_path=os.path.join(output_dir, f"{base_name}_complete_pipeline.png")
        )
        
        # 2. 保存置信度分析
        if results['optimized_detections']['lines'] or results['optimized_detections']['endpoints']:
            self.visualizer.visualize_confidence_analysis(
                results['optimized_detections']['lines'],
                results['optimized_detections']['endpoints'],
                save_path=os.path.join(output_dir, f"{base_name}_confidence_analysis.png")
            )
        
        # 3. 保存优化对比（如果使用了强化学习）
        if self.use_rl_optimization:
            self.visualizer.visualize_detection_comparison(
                results['original_image'],
                results['initial_detections'],
                results['optimized_detections'],
                save_path=os.path.join(output_dir, f"{base_name}_optimization_comparison.png")
            )
        
        # 4. 保存热力图叠加
        self.visualizer.visualize_heatmap_overlay(
            results['original_image'],
            results['cnn_outputs']['line_heatmap'],
            results['cnn_outputs']['endpoint_heatmap'],
            save_path=os.path.join(output_dir, f"{base_name}_heatmap_overlay.png")
        )
        
        # 5. 保存坐标数据
        coordinates_file = os.path.join(output_dir, f"{base_name}_coordinates.txt")
        self._save_coordinates_to_file(results['coordinates'], coordinates_file)
        
        print(f"结果已保存到: {output_dir}")
    
    def _save_coordinates_to_file(self, coordinates: Dict[str, List], filepath: str):
        """保存坐标到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("几何检测结果坐标\n")
            f.write("="*50 + "\n\n")
            
            # 线条坐标
            f.write(f"线条坐标 ({len(coordinates['line_coordinates'])} 条):\n")
            f.write("-" * 30 + "\n")
            for i, line in enumerate(coordinates['line_coordinates']):
                conf = coordinates['line_confidences'][i] if i < len(coordinates['line_confidences']) else 0
                f.write(f"线条 {i+1}: ({line[0]}, {line[1]}) -> ({line[2]}, {line[3]}) "
                       f"置信度: {conf:.3f}\n")
            
            f.write("\n")
            
            # 端点坐标
            f.write(f"端点坐标 ({len(coordinates['endpoint_coordinates'])} 个):\n")
            f.write("-" * 30 + "\n")
            for i, endpoint in enumerate(coordinates['endpoint_coordinates']):
                conf = coordinates['endpoint_confidences'][i] if i < len(coordinates['endpoint_confidences']) else 0
                ep_type = coordinates['endpoint_types'][i] if i < len(coordinates['endpoint_types']) else 'unknown'
                f.write(f"端点 {i+1}: ({endpoint[0]}, {endpoint[1]}) "
                       f"类型: {ep_type} 置信度: {conf:.3f}\n")
    
    def batch_process(self, 
                     input_dir: str, 
                     output_dir: str = "batch_results",
                     image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']) -> List[Dict[str, Any]]:
        """
        批量处理图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            image_extensions: 支持的图像扩展名
            
        Returns:
            所有处理结果的列表
        """
        print(f"开始批量处理目录: {input_dir}")
        
        # 查找图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 处理每个图像
        all_results = []
        for i, image_file in enumerate(image_files):
            print(f"\n处理进度: {i+1}/{len(image_files)}")
            try:
                results = self.detect_geometry(
                    str(image_file), 
                    save_results=True,
                    output_dir=os.path.join(output_dir, image_file.stem)
                )
                all_results.append(results)
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
        
        print(f"\n批量处理完成！成功处理 {len(all_results)} 个文件")
        return all_results
    
    def train_rl_optimizer(self, 
                          training_steps: int = 50000,
                          save_model: bool = True,
                          model_save_path: str = "models/rl_optimizer.zip"):
        """
        训练强化学习优化器
        
        Args:
            training_steps: 训练步数
            save_model: 是否保存模型
            model_save_path: 模型保存路径
        """
        if not self.use_rl_optimization:
            print("强化学习优化器未启用")
            return
        
        print(f"开始训练强化学习优化器，训练步数: {training_steps}")
        
        # 训练
        self.rl_optimizer.train(total_timesteps=training_steps)
        
        # 保存模型
        if save_model:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            self.rl_optimizer.save_model(model_save_path)
        
        print("强化学习优化器训练完成！")


def create_sample_geometry_image(save_path: str = "sample_geometry.png"):
    """创建示例几何图像"""
    # 创建空白图像
    img = np.zeros((512, 512), dtype=np.uint8)
    
    # 绘制三角形
    triangle_points = np.array([[100, 400], [200, 100], [350, 380]], np.int32)
    cv2.polylines(img, [triangle_points], True, 255, 3)
    
    # 绘制矩形
    cv2.rectangle(img, (50, 50), (200, 150), 255, 2)
    
    # 绘制几条直线
    cv2.line(img, (300, 50), (450, 200), 255, 2)
    cv2.line(img, (400, 300), (480, 450), 255, 2)
    cv2.line(img, (50, 300), (250, 250), 255, 2)
    
    # 添加一些噪声
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # 保存图像
    cv2.imwrite(save_path, img)
    print(f"示例几何图像已保存到: {save_path}")
    
    return save_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="几何线条和端点检测系统")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--batch", type=str, help="批量处理目录")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "unet"], 
                       help="CNN模型类型")
    parser.add_argument("--no-rl", action="store_true", help="禁用强化学习优化")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    parser.add_argument("--train-rl", action="store_true", help="训练强化学习优化器")
    parser.add_argument("--train-steps", type=int, default=50000, help="强化学习训练步数")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    
    args = parser.parse_args()
    
    # 创建检测管道
    pipeline = GeometryDetectionPipeline(
        model_type=args.model,
        use_rl_optimization=not args.no_rl,
        device=args.device
    )
    
    try:
        if args.demo:
            # 演示模式
            print("运行演示模式...")
            sample_image = create_sample_geometry_image()
            results = pipeline.detect_geometry(sample_image, save_results=True, output_dir=args.output)
            
            print("\n=== 演示结果 ===")
            print(f"检测到线条数量: {len(results['optimized_detections']['lines'])}")
            print(f"检测到端点数量: {len(results['optimized_detections']['endpoints'])}")
            print(f"处理时间: {results['processing_time']:.2f}秒")
            
        elif args.train_rl:
            # 训练强化学习模式
            pipeline.train_rl_optimizer(
                training_steps=args.train_steps,
                save_model=True
            )
            
        elif args.image:
            # 单图像处理
            results = pipeline.detect_geometry(args.image, save_results=True, output_dir=args.output)
            
            print("\n=== 检测结果 ===")
            print(f"检测到线条数量: {len(results['optimized_detections']['lines'])}")
            print(f"检测到端点数量: {len(results['optimized_detections']['endpoints'])}")
            print(f"处理时间: {results['processing_time']:.2f}秒")
            
        elif args.batch:
            # 批量处理
            all_results = pipeline.batch_process(args.batch, args.output)
            
            print("\n=== 批量处理统计 ===")
            total_lines = sum(len(r['optimized_detections']['lines']) for r in all_results)
            total_endpoints = sum(len(r['optimized_detections']['endpoints']) for r in all_results)
            total_time = sum(r['processing_time'] for r in all_results)
            
            print(f"处理文件数量: {len(all_results)}")
            print(f"总检测线条数: {total_lines}")
            print(f"总检测端点数: {total_endpoints}")
            print(f"总处理时间: {total_time:.2f}秒")
            print(f"平均处理时间: {total_time/len(all_results):.2f}秒/图像")
            
        else:
            # 显示帮助信息
            parser.print_help()
            print("\n示例用法:")
            print("  python main.py --demo                    # 运行演示")
            print("  python main.py --image sample.png        # 处理单个图像")
            print("  python main.py --batch ./images          # 批量处理")
            print("  python main.py --train-rl                # 训练强化学习优化器")
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()