import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from postprocessing import LineSegment, EndPoint
import os


class GeometryVisualizer:
    """几何检测结果的可视化工具"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        初始化可视化器
        
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize
        self.colors = {
            'line': '#FF6B6B',      # 红色
            'endpoint': '#4ECDC4',   # 青色
            'corner': '#45B7D1',     # 蓝色
            'intersection': '#96CEB4', # 绿色
            'terminus': '#FECA57',   # 黄色
            'background': '#2C3E50'  # 深蓝灰色
        }
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_complete_pipeline(self, 
                                  original_image: np.ndarray,
                                  line_heatmap: np.ndarray,
                                  endpoint_heatmap: np.ndarray,
                                  detected_lines: List[LineSegment],
                                  detected_endpoints: List[EndPoint],
                                  save_path: Optional[str] = None) -> None:
        """
        可视化完整的检测管道
        
        Args:
            original_image: 原始图像
            line_heatmap: 线条热力图
            endpoint_heatmap: 端点热力图
            detected_lines: 检测到的线条
            detected_endpoints: 检测到的端点
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('几何检测完整管道结果', fontsize=16, fontweight='bold')
        
        # 1. 原始图像
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 2. 线条热力图
        im1 = axes[0, 1].imshow(line_heatmap, cmap='hot', alpha=0.8)
        axes[0, 1].set_title('线条热力图')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. 端点热力图
        im2 = axes[0, 2].imshow(endpoint_heatmap, cmap='viridis', alpha=0.8)
        axes[0, 2].set_title('端点热力图')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. 检测到的线条
        axes[1, 0].imshow(original_image, cmap='gray', alpha=0.7)
        self._draw_lines(axes[1, 0], detected_lines)
        axes[1, 0].set_title(f'检测到的线条 ({len(detected_lines)})')
        axes[1, 0].axis('off')
        
        # 5. 检测到的端点
        axes[1, 1].imshow(original_image, cmap='gray', alpha=0.7)
        self._draw_endpoints(axes[1, 1], detected_endpoints)
        axes[1, 1].set_title(f'检测到的端点 ({len(detected_endpoints)})')
        axes[1, 1].axis('off')
        
        # 6. 综合结果
        axes[1, 2].imshow(original_image, cmap='gray', alpha=0.7)
        self._draw_lines(axes[1, 2], detected_lines)
        self._draw_endpoints(axes[1, 2], detected_endpoints)
        axes[1, 2].set_title('综合检测结果')
        axes[1, 2].axis('off')
        
        # 添加图例
        self._add_legend(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def visualize_detection_comparison(self,
                                     original_image: np.ndarray,
                                     before_optimization: Dict[str, List],
                                     after_optimization: Dict[str, List],
                                     save_path: Optional[str] = None) -> None:
        """
        可视化优化前后的对比
        
        Args:
            original_image: 原始图像
            before_optimization: 优化前的检测结果
            after_optimization: 优化后的检测结果
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('强化学习优化前后对比', fontsize=16, fontweight='bold')
        
        # 原始图像
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 优化前
        axes[1].imshow(original_image, cmap='gray', alpha=0.7)
        self._draw_detection_dict(axes[1], before_optimization)
        axes[1].set_title(f'优化前\n线条: {len(before_optimization.get("lines", []))}, '
                         f'端点: {len(before_optimization.get("endpoints", []))}')
        axes[1].axis('off')
        
        # 优化后
        axes[2].imshow(original_image, cmap='gray', alpha=0.7)
        self._draw_detection_dict(axes[2], after_optimization)
        axes[2].set_title(f'优化后\n线条: {len(after_optimization.get("lines", []))}, '
                         f'端点: {len(after_optimization.get("endpoints", []))}')
        axes[2].axis('off')
        
        # 添加图例
        self._add_legend(axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比结果已保存到: {save_path}")
        
        plt.show()
    
    def visualize_confidence_analysis(self,
                                    detected_lines: List[LineSegment],
                                    detected_endpoints: List[EndPoint],
                                    save_path: Optional[str] = None) -> None:
        """
        可视化置信度分析
        
        Args:
            detected_lines: 检测到的线条
            detected_endpoints: 检测到的端点
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('检测置信度分析', fontsize=16, fontweight='bold')
        
        # 线条置信度分布
        if detected_lines:
            line_confidences = [line.confidence for line in detected_lines]
            axes[0, 0].hist(line_confidences, bins=20, alpha=0.7, color=self.colors['line'])
            axes[0, 0].set_title('线条置信度分布')
            axes[0, 0].set_xlabel('置信度')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].axvline(np.mean(line_confidences), color='red', 
                              linestyle='--', label=f'平均值: {np.mean(line_confidences):.3f}')
            axes[0, 0].legend()
        
        # 端点置信度分布
        if detected_endpoints:
            endpoint_confidences = [ep.confidence for ep in detected_endpoints]
            axes[0, 1].hist(endpoint_confidences, bins=20, alpha=0.7, color=self.colors['endpoint'])
            axes[0, 1].set_title('端点置信度分布')
            axes[0, 1].set_xlabel('置信度')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].axvline(np.mean(endpoint_confidences), color='red', 
                              linestyle='--', label=f'平均值: {np.mean(endpoint_confidences):.3f}')
            axes[0, 1].legend()
        
        # 线条长度与置信度关系
        if detected_lines:
            lengths = [line.length for line in detected_lines]
            confidences = [line.confidence for line in detected_lines]
            axes[1, 0].scatter(lengths, confidences, alpha=0.6, color=self.colors['line'])
            axes[1, 0].set_title('线条长度 vs 置信度')
            axes[1, 0].set_xlabel('线条长度')
            axes[1, 0].set_ylabel('置信度')
            
            # 添加趋势线
            if len(lengths) > 1:
                z = np.polyfit(lengths, confidences, 1)
                p = np.poly1d(z)
                axes[1, 0].plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8)
        
        # 端点类型分布
        if detected_endpoints:
            endpoint_types = [ep.point_type for ep in detected_endpoints]
            type_counts = {}
            for ep_type in endpoint_types:
                type_counts[ep_type] = type_counts.get(ep_type, 0) + 1
            
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = [self.colors.get(t, '#95A5A6') for t in types]
            
            axes[1, 1].pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('端点类型分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"置信度分析已保存到: {save_path}")
        
        plt.show()
    
    def create_training_animation(self,
                                training_images: List[np.ndarray],
                                training_results: List[Dict],
                                save_path: str) -> None:
        """
        创建训练过程的动画
        
        Args:
            training_images: 训练图像列表
            training_results: 训练结果列表
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            if frame < len(training_images):
                image = training_images[frame]
                results = training_results[frame]
                
                ax.imshow(image, cmap='gray', alpha=0.7)
                self._draw_detection_dict(ax, results)
                ax.set_title(f'训练步骤 {frame + 1}')
                ax.axis('off')
        
        ani = animation.FuncAnimation(fig, animate, frames=len(training_images), 
                                    interval=500, repeat=True)
        
        # 保存动画
        ani.save(save_path, writer='pillow', fps=2)
        print(f"训练动画已保存到: {save_path}")
        
        plt.show()
    
    def visualize_heatmap_overlay(self,
                                original_image: np.ndarray,
                                line_heatmap: np.ndarray,
                                endpoint_heatmap: np.ndarray,
                                alpha: float = 0.6,
                                save_path: Optional[str] = None) -> None:
        """
        可视化热力图叠加效果
        
        Args:
            original_image: 原始图像
            line_heatmap: 线条热力图
            endpoint_heatmap: 端点热力图
            alpha: 透明度
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('热力图叠加可视化', fontsize=16, fontweight='bold')
        
        # 线条热力图叠加
        axes[0].imshow(original_image, cmap='gray')
        im1 = axes[0].imshow(line_heatmap, cmap='Reds', alpha=alpha)
        axes[0].set_title('线条热力图叠加')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 端点热力图叠加
        axes[1].imshow(original_image, cmap='gray')
        im2 = axes[1].imshow(endpoint_heatmap, cmap='Blues', alpha=alpha)
        axes[1].set_title('端点热力图叠加')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 组合热力图
        axes[2].imshow(original_image, cmap='gray')
        axes[2].imshow(line_heatmap, cmap='Reds', alpha=alpha * 0.7)
        axes[2].imshow(endpoint_heatmap, cmap='Blues', alpha=alpha * 0.7)
        axes[2].set_title('组合热力图')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图叠加结果已保存到: {save_path}")
        
        plt.show()
    
    def _draw_lines(self, ax, lines: List[LineSegment]) -> None:
        """绘制线条"""
        for line in lines:
            # 根据置信度调整线条粗细和透明度
            linewidth = max(1, int(line.confidence * 5))
            alpha = max(0.3, line.confidence)
            
            ax.plot([line.start_point[0], line.end_point[0]], 
                   [line.start_point[1], line.end_point[1]], 
                   color=self.colors['line'], 
                   linewidth=linewidth, 
                   alpha=alpha)
    
    def _draw_endpoints(self, ax, endpoints: List[EndPoint]) -> None:
        """绘制端点"""
        for endpoint in endpoints:
            # 根据端点类型选择颜色和形状
            color = self.colors.get(endpoint.point_type, self.colors['endpoint'])
            size = max(20, int(endpoint.confidence * 100))
            
            if endpoint.point_type == 'corner':
                marker = 's'  # 方形
            elif endpoint.point_type == 'intersection':
                marker = '*'  # 星形
            elif endpoint.point_type == 'terminus':
                marker = '^'  # 三角形
            else:
                marker = 'o'  # 圆形
            
            ax.scatter(endpoint.position[0], endpoint.position[1], 
                      c=color, s=size, marker=marker, 
                      alpha=max(0.3, endpoint.confidence),
                      edgecolors='white', linewidth=1)
    
    def _draw_detection_dict(self, ax, detections: Dict[str, List]) -> None:
        """绘制检测字典格式的结果"""
        # 处理线条
        if 'lines' in detections:
            lines = detections['lines']
            if lines and isinstance(lines[0], LineSegment):
                self._draw_lines(ax, lines)
            elif lines and isinstance(lines[0], list):
                # 处理坐标格式的线条
                for line_coords in lines:
                    if len(line_coords) >= 4:
                        ax.plot([line_coords[0], line_coords[2]], 
                               [line_coords[1], line_coords[3]], 
                               color=self.colors['line'], linewidth=2)
        
        # 处理端点
        if 'endpoints' in detections:
            endpoints = detections['endpoints']
            if endpoints and isinstance(endpoints[0], EndPoint):
                self._draw_endpoints(ax, endpoints)
            elif endpoints and isinstance(endpoints[0], (tuple, list)):
                # 处理坐标格式的端点
                for endpoint_pos in endpoints:
                    if len(endpoint_pos) >= 2:
                        ax.scatter(endpoint_pos[0], endpoint_pos[1], 
                                 c=self.colors['endpoint'], s=50, marker='o',
                                 edgecolors='white', linewidth=1)
    
    def _add_legend(self, ax) -> None:
        """添加图例"""
        legend_elements = [
            mpatches.Patch(color=self.colors['line'], label='线条'),
            mpatches.Patch(color=self.colors['endpoint'], label='端点'),
            mpatches.Patch(color=self.colors['corner'], label='角点'),
            mpatches.Patch(color=self.colors['intersection'], label='交点'),
            mpatches.Patch(color=self.colors['terminus'], label='端点')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), fontsize=8)
    
    def save_results_summary(self,
                           detection_results: Dict[str, Any],
                           save_dir: str) -> None:
        """
        保存检测结果摘要
        
        Args:
            detection_results: 检测结果
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建摘要图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('检测结果摘要', fontsize=16, fontweight='bold')
        
        # 统计信息
        stats_text = f"""
        检测统计:
        - 线条数量: {len(detection_results.get('lines', []))}
        - 端点数量: {len(detection_results.get('endpoints', []))}
        - 平均线条置信度: {np.mean([l.confidence for l in detection_results.get('lines', [])]) if detection_results.get('lines') else 0:.3f}
        - 平均端点置信度: {np.mean([e.confidence for e in detection_results.get('endpoints', [])]) if detection_results.get('endpoints') else 0:.3f}
        """
        
        axes[0, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[0, 0].set_title('统计信息')
        axes[0, 0].axis('off')
        
        # 其他统计图表可以在这里添加...
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"结果摘要已保存到: {save_dir}/summary.png")


def create_visualizer() -> GeometryVisualizer:
    """创建可视化器的工厂函数"""
    return GeometryVisualizer()


if __name__ == "__main__":
    # 测试可视化器
    visualizer = create_visualizer()
    
    # 创建测试数据
    test_image = np.random.rand(512, 512) * 255
    test_line_heatmap = np.random.rand(512, 512)
    test_endpoint_heatmap = np.random.rand(512, 512)
    
    print("可视化工具创建完成！")
    print("可用功能:")
    print("- visualize_complete_pipeline(): 完整管道可视化")
    print("- visualize_detection_comparison(): 优化前后对比")
    print("- visualize_confidence_analysis(): 置信度分析")
    print("- visualize_heatmap_overlay(): 热力图叠加")
    print("- create_training_animation(): 训练动画")