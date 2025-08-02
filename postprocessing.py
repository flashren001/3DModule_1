import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class LineSegment:
    """线段数据结构"""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    confidence: float
    length: float
    angle: float


@dataclass
class EndPoint:
    """端点数据结构"""
    position: Tuple[int, int]
    confidence: float
    point_type: str  # 'corner', 'intersection', 'terminus'


class GeometryPostProcessor:
    """几何检测的后处理器"""
    
    def __init__(self, 
                 line_threshold: float = 0.5,
                 endpoint_threshold: float = 0.5,
                 min_line_length: float = 20.0,
                 max_line_gap: float = 10.0,
                 angle_threshold: float = 10.0):
        """
        初始化后处理器
        
        Args:
            line_threshold: 线条热力图阈值
            endpoint_threshold: 端点热力图阈值
            min_line_length: 最小线条长度
            max_line_gap: 最大线条间隙
            angle_threshold: 角度合并阈值（度）
        """
        self.line_threshold = line_threshold
        self.endpoint_threshold = endpoint_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_threshold = angle_threshold
    
    def process_detections(self, 
                          line_heatmap: np.ndarray, 
                          endpoint_heatmap: np.ndarray) -> Dict[str, List]:
        """
        主要处理函数，从热力图中提取精确的几何元素
        
        Args:
            line_heatmap: 线条热力图
            endpoint_heatmap: 端点热力图
            
        Returns:
            包含处理后线条和端点的字典
        """
        # 1. 提取线条
        line_segments = self._extract_line_segments(line_heatmap)
        
        # 2. 提取端点
        endpoints = self._extract_endpoints(endpoint_heatmap)
        
        # 3. 线条优化
        optimized_lines = self._optimize_lines(line_segments)
        
        # 4. 端点优化
        optimized_endpoints = self._optimize_endpoints(endpoints, optimized_lines)
        
        # 5. 几何关系验证
        validated_results = self._validate_geometry(optimized_lines, optimized_endpoints)
        
        return validated_results
    
    def _extract_line_segments(self, heatmap: np.ndarray) -> List[LineSegment]:
        """从线条热力图中提取线段"""
        # 阈值化
        binary = (heatmap > self.line_threshold).astype(np.uint8) * 255
        
        # 形态学操作 - 连接断开的线条
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 细化线条
        binary = self._skeletonize(binary)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=int(self.min_line_length * 0.6),
            minLineLength=int(self.min_line_length),
            maxLineGap=int(self.max_line_gap)
        )
        
        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线段属性
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # 计算置信度（基于热力图值）
                confidence = self._calculate_line_confidence(heatmap, (x1, y1), (x2, y2))
                
                line_segments.append(LineSegment(
                    start_point=(x1, y1),
                    end_point=(x2, y2),
                    confidence=confidence,
                    length=length,
                    angle=angle
                ))
        
        return line_segments
    
    def _extract_endpoints(self, heatmap: np.ndarray) -> List[EndPoint]:
        """从端点热力图中提取端点"""
        # 阈值化
        binary = (heatmap > self.endpoint_threshold).astype(np.uint8) * 255
        
        # 非最大值抑制
        local_maxima = self._non_maximum_suppression(heatmap, window_size=5)
        
        # 结合阈值化和局部最大值
        candidate_points = binary & local_maxima
        
        # 查找连通组件
        contours, _ = cv2.findContours(candidate_points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        endpoints = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                # 计算质心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 计算置信度
                    confidence = heatmap[cy, cx] if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1] else 0
                    
                    endpoints.append(EndPoint(
                        position=(cx, cy),
                        confidence=float(confidence),
                        point_type='detected'
                    ))
        
        return endpoints
    
    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """线条细化/骨架化"""
        # 使用OpenCV的细化算法
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # 迭代细化
        skeleton = binary_image.copy()
        while True:
            eroded = cv2.erode(skeleton, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            
            if cv2.countNonZero(cv2.subtract(skeleton, eroded)) == 0:
                break
        
        return skeleton
    
    def _non_maximum_suppression(self, heatmap: np.ndarray, window_size: int = 5) -> np.ndarray:
        """非最大值抑制"""
        # 使用滑动窗口找到局部最大值
        local_maxima = np.zeros_like(heatmap, dtype=np.uint8)
        
        half_window = window_size // 2
        for i in range(half_window, heatmap.shape[0] - half_window):
            for j in range(half_window, heatmap.shape[1] - half_window):
                window = heatmap[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                if heatmap[i, j] == np.max(window) and heatmap[i, j] > 0:
                    local_maxima[i, j] = 255
        
        return local_maxima
    
    def _calculate_line_confidence(self, heatmap: np.ndarray, 
                                 start: Tuple[int, int], 
                                 end: Tuple[int, int]) -> float:
        """计算线条的置信度"""
        # 在线条路径上采样点
        num_points = max(int(np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)), 10)
        x_points = np.linspace(start[0], end[0], num_points).astype(int)
        y_points = np.linspace(start[1], end[1], num_points).astype(int)
        
        # 确保坐标在图像范围内
        valid_mask = (x_points >= 0) & (x_points < heatmap.shape[1]) & \
                    (y_points >= 0) & (y_points < heatmap.shape[0])
        
        if np.sum(valid_mask) == 0:
            return 0.0
        
        # 计算路径上点的平均置信度
        confidence_values = heatmap[y_points[valid_mask], x_points[valid_mask]]
        return np.mean(confidence_values)
    
    def _optimize_lines(self, line_segments: List[LineSegment]) -> List[LineSegment]:
        """优化线条检测结果"""
        if not line_segments:
            return []
        
        # 1. 按置信度排序
        line_segments.sort(key=lambda x: x.confidence, reverse=True)
        
        # 2. 合并相似的线条
        merged_lines = self._merge_similar_lines(line_segments)
        
        # 3. 延长短线条
        extended_lines = self._extend_short_lines(merged_lines)
        
        # 4. 过滤低质量线条
        filtered_lines = [line for line in extended_lines if 
                         line.confidence > 0.3 and line.length > self.min_line_length]
        
        return filtered_lines
    
    def _merge_similar_lines(self, line_segments: List[LineSegment]) -> List[LineSegment]:
        """合并相似的线条"""
        if len(line_segments) <= 1:
            return line_segments
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(line_segments):
            if i in used:
                continue
            
            candidates = [line1]
            
            for j, line2 in enumerate(line_segments[i+1:], i+1):
                if j in used:
                    continue
                
                # 检查角度相似性
                angle_diff = abs(line1.angle - line2.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff < self.angle_threshold:
                    # 检查距离相似性
                    if self._lines_are_collinear(line1, line2):
                        candidates.append(line2)
                        used.add(j)
            
            # 合并候选线条
            if len(candidates) > 1:
                merged_line = self._merge_line_group(candidates)
                merged.append(merged_line)
            else:
                merged.append(line1)
            
            used.add(i)
        
        return merged
    
    def _lines_are_collinear(self, line1: LineSegment, line2: LineSegment, 
                           distance_threshold: float = 15.0) -> bool:
        """检查两条线是否共线"""
        # 计算line1的端点到line2的距离
        distances = [
            self._point_to_line_distance(line1.start_point, line2.start_point, line2.end_point),
            self._point_to_line_distance(line1.end_point, line2.start_point, line2.end_point)
        ]
        
        return all(d < distance_threshold for d in distances)
    
    def _point_to_line_distance(self, point: Tuple[int, int], 
                              line_start: Tuple[int, int], 
                              line_end: Tuple[int, int]) -> float:
        """计算点到线的距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 处理退化情况
        if x1 == x2 and y1 == y2:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # 使用点到直线距离公式
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        return distance
    
    def _merge_line_group(self, lines: List[LineSegment]) -> LineSegment:
        """合并一组线条"""
        # 收集所有端点
        all_points = []
        total_confidence = 0
        
        for line in lines:
            all_points.extend([line.start_point, line.end_point])
            total_confidence += line.confidence
        
        # 找到最远的两个点作为合并后的端点
        max_distance = 0
        best_start, best_end = all_points[0], all_points[1]
        
        for i, p1 in enumerate(all_points):
            for j, p2 in enumerate(all_points[i+1:], i+1):
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if distance > max_distance:
                    max_distance = distance
                    best_start, best_end = p1, p2
        
        # 计算新线条的属性
        length = np.sqrt((best_end[0] - best_start[0])**2 + (best_end[1] - best_start[1])**2)
        angle = np.degrees(np.arctan2(best_end[1] - best_start[1], best_end[0] - best_start[0]))
        confidence = total_confidence / len(lines)
        
        return LineSegment(
            start_point=best_start,
            end_point=best_end,
            confidence=confidence,
            length=length,
            angle=angle
        )
    
    def _extend_short_lines(self, line_segments: List[LineSegment]) -> List[LineSegment]:
        """延长短线条"""
        extended = []
        
        for line in line_segments:
            if line.length < self.min_line_length * 1.5:
                # 尝试延长线条
                extended_line = self._extend_line(line, line_segments)
                extended.append(extended_line)
            else:
                extended.append(line)
        
        return extended
    
    def _extend_line(self, target_line: LineSegment, 
                    all_lines: List[LineSegment]) -> LineSegment:
        """延长单条线条"""
        # 这里可以实现线条延长的逻辑
        # 暂时返回原线条
        return target_line
    
    def _optimize_endpoints(self, endpoints: List[EndPoint], 
                          lines: List[LineSegment]) -> List[EndPoint]:
        """优化端点检测结果"""
        if not endpoints:
            return []
        
        # 1. 聚类相近的端点
        clustered_endpoints = self._cluster_endpoints(endpoints)
        
        # 2. 分类端点类型
        classified_endpoints = self._classify_endpoint_types(clustered_endpoints, lines)
        
        # 3. 过滤低质量端点
        filtered_endpoints = [ep for ep in classified_endpoints if ep.confidence > 0.3]
        
        return filtered_endpoints
    
    def _cluster_endpoints(self, endpoints: List[EndPoint], 
                          cluster_distance: float = 10.0) -> List[EndPoint]:
        """聚类相近的端点"""
        if len(endpoints) <= 1:
            return endpoints
        
        # 提取位置信息
        positions = np.array([ep.position for ep in endpoints])
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(positions)
        
        clustered = []
        for cluster_id in set(clustering.labels_):
            cluster_mask = clustering.labels_ == cluster_id
            cluster_endpoints = [endpoints[i] for i in range(len(endpoints)) if cluster_mask[i]]
            
            # 合并聚类中的端点
            merged_endpoint = self._merge_endpoint_cluster(cluster_endpoints)
            clustered.append(merged_endpoint)
        
        return clustered
    
    def _merge_endpoint_cluster(self, cluster: List[EndPoint]) -> EndPoint:
        """合并一个端点聚类"""
        # 计算平均位置
        avg_x = int(np.mean([ep.position[0] for ep in cluster]))
        avg_y = int(np.mean([ep.position[1] for ep in cluster]))
        
        # 计算平均置信度
        avg_confidence = np.mean([ep.confidence for ep in cluster])
        
        return EndPoint(
            position=(avg_x, avg_y),
            confidence=avg_confidence,
            point_type='merged'
        )
    
    def _classify_endpoint_types(self, endpoints: List[EndPoint], 
                               lines: List[LineSegment]) -> List[EndPoint]:
        """分类端点类型"""
        classified = []
        
        for endpoint in endpoints:
            # 计算该端点附近的线条数量
            nearby_lines = self._count_nearby_lines(endpoint, lines)
            
            # 根据附近线条数量分类
            if nearby_lines == 0:
                point_type = 'isolated'
            elif nearby_lines == 1:
                point_type = 'terminus'
            elif nearby_lines == 2:
                point_type = 'corner'
            else:
                point_type = 'intersection'
            
            classified.append(EndPoint(
                position=endpoint.position,
                confidence=endpoint.confidence,
                point_type=point_type
            ))
        
        return classified
    
    def _count_nearby_lines(self, endpoint: EndPoint, lines: List[LineSegment], 
                          distance_threshold: float = 15.0) -> int:
        """计算端点附近的线条数量"""
        count = 0
        for line in lines:
            # 计算端点到线条的距离
            distance = min(
                self._point_to_line_distance(endpoint.position, line.start_point, line.end_point),
                np.sqrt((endpoint.position[0] - line.start_point[0])**2 + 
                       (endpoint.position[1] - line.start_point[1])**2),
                np.sqrt((endpoint.position[0] - line.end_point[0])**2 + 
                       (endpoint.position[1] - line.end_point[1])**2)
            )
            
            if distance < distance_threshold:
                count += 1
        
        return count
    
    def _validate_geometry(self, lines: List[LineSegment], 
                          endpoints: List[EndPoint]) -> Dict[str, List]:
        """验证几何关系的一致性"""
        # 验证端点是否在线条端点附近
        validated_endpoints = []
        
        for endpoint in endpoints:
            is_valid = False
            
            # 检查是否在任何线条的端点附近
            for line in lines:
                for line_point in [line.start_point, line.end_point]:
                    distance = np.sqrt((endpoint.position[0] - line_point[0])**2 + 
                                     (endpoint.position[1] - line_point[1])**2)
                    if distance < 20:  # 阈值
                        is_valid = True
                        break
                
                if is_valid:
                    break
            
            # 也检查是否在线条上
            if not is_valid:
                for line in lines:
                    distance = self._point_to_line_distance(
                        endpoint.position, line.start_point, line.end_point)
                    if distance < 10:  # 更严格的阈值
                        is_valid = True
                        break
            
            if is_valid:
                validated_endpoints.append(endpoint)
        
        return {
            'lines': lines,
            'endpoints': validated_endpoints
        }


def convert_to_coordinates(results: Dict[str, List]) -> Dict[str, List]:
    """将处理结果转换为坐标格式"""
    line_coords = []
    endpoint_coords = []
    
    for line in results['lines']:
        line_coords.append([
            line.start_point[0], line.start_point[1],
            line.end_point[0], line.end_point[1]
        ])
    
    for endpoint in results['endpoints']:
        endpoint_coords.append(endpoint.position)
    
    return {
        'line_coordinates': line_coords,
        'endpoint_coordinates': endpoint_coords,
        'line_confidences': [line.confidence for line in results['lines']],
        'endpoint_confidences': [ep.confidence for ep in results['endpoints']],
        'endpoint_types': [ep.point_type for ep in results['endpoints']]
    }


if __name__ == "__main__":
    # 测试后处理器
    processor = GeometryPostProcessor()
    
    # 创建测试数据
    test_line_heatmap = np.random.rand(512, 512) * 0.3
    test_endpoint_heatmap = np.random.rand(512, 512) * 0.3
    
    # 添加一些明显的特征
    test_line_heatmap[100:200, 100:300] = 0.8  # 水平线
    test_line_heatmap[100:400, 200:210] = 0.9  # 竖直线
    
    test_endpoint_heatmap[100, 100] = 0.9  # 端点
    test_endpoint_heatmap[200, 300] = 0.8  # 端点
    
    # 处理
    results = processor.process_detections(test_line_heatmap, test_endpoint_heatmap)
    coordinates = convert_to_coordinates(results)
    
    print(f"检测到 {len(results['lines'])} 条线条")
    print(f"检测到 {len(results['endpoints'])} 个端点")
    print("后处理模块创建完成！")