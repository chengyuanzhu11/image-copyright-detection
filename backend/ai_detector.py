import os
import numpy as np
import time
import logging
from PIL import Image
import cv2
import random
from typing import Tuple, Dict, Any, List, Union, Optional
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetector:
    
    def __init__(self):
        self.model_loaded = False
        try:
            self.model_loaded = True
            logger.info("AI检测器初始化成功")
        except Exception as e:
            logger.error(f"AI检测器初始化失败: {e}")
    
    def detect(self, image_path: str) -> Tuple[float, Dict[str, Any]]:

        start_time = time.time()
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            analysis = self._analyze_image(img, img_array)
            
            noise_analysis = self._analyze_noise(img_array)
            analysis['noise_analysis'] = noise_analysis
            
            texture_analysis = self._analyze_texture(img_array)
            analysis['texture_analysis'] = texture_analysis
            
            edge_analysis = self._analyze_edges(img_array)
            analysis['edge_analysis'] = edge_analysis
            
            color_analysis = self._analyze_colors(img_array)
            analysis['color_analysis'] = color_analysis
            
            feature_stats = self._get_feature_statistics(analysis)
            analysis['feature_statistics'] = feature_stats
            
            is_id_photo = self._check_if_id_photo(img_array, analysis)
            analysis['is_id_photo'] = is_id_photo
            
            score = self._calculate_ai_score(analysis)
            
            if is_id_photo and score > 0.35:
                score = max(0.1, score * 0.7) 
                
            analysis['suspicious_features'] = self._identify_suspicious_features(analysis, score)
            
            threshold = float(os.environ.get('AI_DETECTION_THRESHOLD', '0.5'))
            analysis['conclusion'] = 'AI生成' if score > threshold else '真实照片'
            analysis['ai_generated_probability'] = score
            analysis['detector_type'] = 'advanced_feature_based'
            analysis['detection_time'] = time.time() - start_time
            
            score = float(score)
            
            logger.info(f"检测完成: 得分={score:.4f}, 结论={analysis['conclusion']}, 执行时间={analysis['detection_time']:.2f}秒")
            return score, analysis
            
        except Exception as e:
            logger.error(f"AI检测过程发生错误: {e}")
            logger.error(traceback.format_exc())
            return 0.5, {
                'ai_generated_probability': 0.5,
                'conclusion': '无法确定',
                'error': str(e),
                'detection_time': time.time() - start_time,
                'detector_type': 'error_fallback'
            }
    
    def _analyze_image(self, img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """分析图像的基本属性"""
        width, height = img.size
        mode = img.mode
        format_info = getattr(img, 'format', 'Unknown')
        has_alpha = 'A' in mode
        
        # 计算图像的一些基本统计数据
        brightness = np.mean(img_array) / 255.0
        contrast = np.std(img_array) / 255.0
        
        # 检查图像尺寸和比例
        aspect_ratio = width / height if height > 0 else 0
        is_standard_size = False
        
        standard_sizes = [(1024, 1024), (512, 512), (768, 768), (1024, 768), (768, 1024)]
        for std_w, std_h in standard_sizes:
            if (width == std_w and height == std_h) or (width == std_h and height == std_w):
                is_standard_size = True
                break
        
        return {
            'image_info': {
                'width': width,
                'height': height,
                'format': format_info,
                'mode': mode,
                'has_alpha': has_alpha,
                'aspect_ratio': aspect_ratio,
                'is_standard_size': is_standard_size
            },
            'basic_stats': {
                'brightness': float(brightness),
                'contrast': float(contrast)
            }
        }
    
    def _analyze_noise(self, img_array: np.ndarray) -> Dict[str, float]:
        """分析图像噪声模式，AI生成图像通常噪声较少或噪声分布不自然"""
        try:
            # 转换为灰度图像
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                
            # 使用拉普拉斯算子提取噪声
            noise = cv2.Laplacian(gray, cv2.CV_64F)
            
            # 计算噪声统计数据
            noise_mean = np.mean(np.abs(noise))
            noise_std = np.std(noise)
            noise_max = np.max(np.abs(noise))
            
            # 计算噪声分布的偏度和峰度
            noise_flat = noise.flatten()
            noise_skewness = float(np.mean((noise_flat - np.mean(noise_flat))**3) / (np.std(noise_flat)**3)) if np.std(noise_flat) > 0 else 0
            noise_kurtosis = float(np.mean((noise_flat - np.mean(noise_flat))**4) / (np.std(noise_flat)**4)) if np.std(noise_flat) > 0 else 0
            
            return {
                'noise_mean': float(noise_mean),
                'noise_std': float(noise_std),
                'noise_max': float(noise_max),
                'noise_skewness': float(noise_skewness),
                'noise_kurtosis': float(noise_kurtosis)
            }
        except Exception as e:
            logger.warning(f"噪声分析失败: {e}")
            return {
                'noise_mean': 0.0,
                'noise_std': 0.0,
                'noise_max': 0.0,
                'noise_error': str(e)
            }
    
    def _analyze_texture(self, img_array: np.ndarray) -> Dict[str, float]:
        """分析图像纹理，AI生成图像的纹理通常更均匀或有规律"""
        try:
            # 转换为灰度图像
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 通过GLCM(灰度共生矩阵)分析纹理
            # 这里简化处理，使用梯度的方差作为纹理复杂度指标
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            texture_mean = np.mean(gradient_magnitude)
            texture_var = np.var(gradient_magnitude)
            texture_entropy = float(-np.sum(gradient_magnitude * np.log2(gradient_magnitude + 1e-10)) / (img_array.shape[0] * img_array.shape[1]))
            
            return {
                'texture_mean': float(texture_mean),
                'texture_var': float(texture_var),
                'texture_entropy': float(texture_entropy)
            }
        except Exception as e:
            logger.warning(f"纹理分析失败: {e}")
            return {
                'texture_mean': 0.0,
                'texture_var': 0.0,
                'texture_entropy': 0.0,
                'texture_error': str(e)
            }
    
    def _analyze_edges(self, img_array: np.ndarray) -> Dict[str, float]:
        """分析图像边缘，AI生成图像的边缘可能过于完美或不自然"""
        try:
            # 转换为灰度图像
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 计算边缘密度和统计数据
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 计算边缘长度分布（简化）
            edge_mean = np.mean(edges[edges > 0]) if np.sum(edges > 0) > 0 else 0
            edge_std = np.std(edges[edges > 0]) if np.sum(edges > 0) > 0 else 0
            
            return {
                'edge_density': float(edge_density),
                'edge_mean': float(edge_mean),
                'edge_std': float(edge_std)
            }
        except Exception as e:
            logger.warning(f"边缘分析失败: {e}")
            return {
                'edge_density': 0.0,
                'edge_mean': 0.0,
                'edge_std': 0.0,
                'edge_error': str(e)
            }
    
    def _analyze_colors(self, img_array: np.ndarray) -> Dict[str, float]:
        """分析图像颜色分布，AI生成图像的颜色分布可能不同于自然图像"""
        try:
            # 确保图像是RGB格式
            if len(img_array.shape) != 3 or img_array.shape[2] < 3:
                return {'color_error': 'Image is not RGB'}
            
            # 分解RGB通道
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # 计算每个通道的统计数据
            r_mean, r_std = np.mean(r) / 255.0, np.std(r) / 255.0
            g_mean, g_std = np.mean(g) / 255.0, np.std(g) / 255.0
            b_mean, b_std = np.mean(b) / 255.0, np.std(b) / 255.0
            
            # 计算通道间相关性
            rg_corr = np.corrcoef(r.flat, g.flat)[0,1] if np.std(r.flat) > 0 and np.std(g.flat) > 0 else 0
            rb_corr = np.corrcoef(r.flat, b.flat)[0,1] if np.std(r.flat) > 0 and np.std(b.flat) > 0 else 0
            gb_corr = np.corrcoef(g.flat, b.flat)[0,1] if np.std(g.flat) > 0 and np.std(b.flat) > 0 else 0
            
            # 颜色多样性
            colors = img_array.reshape(-1, 3)
            unique_colors = np.unique(colors, axis=0)
            color_diversity = len(unique_colors) / (img_array.shape[0] * img_array.shape[1])
            
            return {
                'r_mean': float(r_mean),
                'r_std': float(r_std),
                'g_mean': float(g_mean),
                'g_std': float(g_std),
                'b_mean': float(b_mean),
                'b_std': float(b_std),
                'rg_correlation': float(rg_corr),
                'rb_correlation': float(rb_corr),
                'gb_correlation': float(gb_corr),
                'color_diversity': float(color_diversity)
            }
        except Exception as e:
            logger.warning(f"颜色分析失败: {e}")
            return {
                'r_mean': 0.0,
                'r_std': 0.0,
                'g_mean': 0.0,
                'g_std': 0.0,
                'b_mean': 0.0,
                'b_std': 0.0,
                'color_error': str(e)
            }
    
    def _get_feature_statistics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """从各种分析中提取特征并计算统计数据"""
        try:
            # 构建特征向量
            features = []
            
            # 添加各种分析中的数值特征
            if 'noise_analysis' in analysis:
                for k, v in analysis['noise_analysis'].items():
                    if isinstance(v, (int, float)) and not k.endswith('_error'):
                        features.append(v)
            
            if 'texture_analysis' in analysis:
                for k, v in analysis['texture_analysis'].items():
                    if isinstance(v, (int, float)) and not k.endswith('_error'):
                        features.append(v)
            
            if 'edge_analysis' in analysis:
                for k, v in analysis['edge_analysis'].items():
                    if isinstance(v, (int, float)) and not k.endswith('_error'):
                        features.append(v)
            
            if 'color_analysis' in analysis:
                for k, v in analysis['color_analysis'].items():
                    if isinstance(v, (int, float)) and not k.endswith('_error'):
                        features.append(v)
            
            # 计算特征向量的统计数据
            if features:
                features = np.array(features)
                return {
                    'mean': float(np.mean(features)),
                    'std': float(np.std(features)),
                    'min': float(np.min(features)),
                    'max': float(np.max(features)),
                    'count': int(len(features))
                }
            else:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
        except Exception as e:
            logger.warning(f"特征统计计算失败: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'error': str(e)
            }
    
    def _check_if_id_photo(self, img_array: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """检查图像是否像证件照，避免误判"""
        try:
            # 特征判断是否为证件照
            # 1. 尺寸比例接近证件照
            aspect_ratio = analysis['image_info']['aspect_ratio']
            standard_id_ratio = 0.7143  # 标准证件照比例 35x45mm
            ratio_diff = abs(aspect_ratio - standard_id_ratio)
            
            # 2. 背景颜色单一
            if 'color_analysis' in analysis:
                color_diversity = analysis['color_analysis'].get('color_diversity', 1.0)
            else:
                color_diversity = 1.0
                
            # 3. 边缘密度低（证件照通常背景与人物分明）
            if 'edge_analysis' in analysis:
                edge_density = analysis['edge_analysis'].get('edge_density', 0.5)
            else:
                edge_density = 0.5
            
            # 综合判断
            if (ratio_diff < 0.1 and color_diversity < 0.2) or \
               (ratio_diff < 0.15 and color_diversity < 0.15 and edge_density < 0.2):
                return True
            
            return False
        except Exception as e:
            logger.warning(f"证件照检查失败: {e}")
            return False
    
    def _calculate_ai_score(self, analysis: Dict[str, Any]) -> float:
        """根据各种分析计算图像为AI生成的概率"""
        try:
            # 基础分数
            score = 0.5
            
            # 特征评分
            if 'noise_analysis' in analysis:
                noise_std = analysis['noise_analysis'].get('noise_std', 20)
                noise_skewness = analysis['noise_analysis'].get('noise_skewness', 0)
                
                # AI生成图像的噪声通常较少且分布更均匀
                if noise_std < 10:
                    score += 0.1
                elif noise_std > 30:
                    score -= 0.1
                
                # AI生成图像的噪声分布偏斜度通常较小
                if abs(noise_skewness) < 0.3:
                    score += 0.05
            
            # 纹理特征评分
            if 'texture_analysis' in analysis:
                texture_var = analysis['texture_analysis'].get('texture_var', 1000)
                texture_entropy = analysis['texture_analysis'].get('texture_entropy', 5)
                
                # AI生成图像的纹理变化通常较小
                if texture_var < 500:
                    score += 0.1
                elif texture_var > 2000:
                    score -= 0.1
                
                # AI生成图像的纹理熵通常较低
                if texture_entropy < 3:
                    score += 0.05
                elif texture_entropy > 6:
                    score -= 0.05
            
            # 边缘特征评分
            if 'edge_analysis' in analysis:
                edge_density = analysis['edge_analysis'].get('edge_density', 0.1)
                
                # AI生成图像的边缘密度通常较低或较高，不太自然
                if edge_density < 0.05 or edge_density > 0.3:
                    score += 0.05
            
            # 颜色特征评分
            if 'color_analysis' in analysis:
                rg_corr = analysis['color_analysis'].get('rg_correlation', 0)
                rb_corr = analysis['color_analysis'].get('rb_correlation', 0)
                gb_corr = analysis['color_analysis'].get('gb_correlation', 0)
                
                # AI生成图像的颜色通道相关性通常较高
                avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
                if avg_corr > 0.85:
                    score += 0.1
            
            # 图像尺寸和比例评分
            if analysis['image_info'].get('is_standard_size', False):
                score += 0.1  # AI生成图像通常是标准尺寸
            
            # 特征统计评分
            if 'feature_statistics' in analysis:
                feature_std = analysis['feature_statistics'].get('std', 0.5)
                
                # AI生成图像的特征分布通常更均匀
                if feature_std < 0.15:
                    score += 0.1
                elif feature_std > 0.5:
                    score -= 0.1
            
            # 限制分数在0-1范围内
            score = max(0.0, min(1.0, score))
            
            return score
        except Exception as e:
            logger.warning(f"AI分数计算失败: {e}")
            return 0.5
    
    def _identify_suspicious_features(self, analysis: Dict[str, Any], score: float) -> List[str]:
        """识别图像中可疑的AI生成特征"""
        suspicious_features = []
        
        try:
            # 噪声特征
            if 'noise_analysis' in analysis:
                noise_std = analysis['noise_analysis'].get('noise_std', 20)
                if noise_std < 10:
                    suspicious_features.append("噪声分布异常均匀")
            
            # 纹理特征
            if 'texture_analysis' in analysis:
                texture_var = analysis['texture_analysis'].get('texture_var', 1000)
                if texture_var < 500:
                    suspicious_features.append("纹理特征不自然")
            
            # 边缘特征
            if 'edge_analysis' in analysis:
                edge_density = analysis['edge_analysis'].get('edge_density', 0.1)
                if edge_density < 0.03:
                    suspicious_features.append("边缘过于平滑")
                elif edge_density > 0.3:
                    suspicious_features.append("边缘分布不自然")
            
            # 颜色特征
            if 'color_analysis' in analysis:
                rg_corr = analysis['color_analysis'].get('rg_correlation', 0)
                rb_corr = analysis['color_analysis'].get('rb_correlation', 0)
                gb_corr = analysis['color_analysis'].get('gb_correlation', 0)
                avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
                if avg_corr > 0.85:
                    suspicious_features.append("颜色分布高度相关")
            
            # 特征统计
            if 'feature_statistics' in analysis:
                feature_std = analysis['feature_statistics'].get('std', 0.5)
                feature_max = analysis['feature_statistics'].get('max', 1.0)
                feature_min = analysis['feature_statistics'].get('min', 0.0)
                
                if feature_std < 0.15:
                    suspicious_features.append("特征分布过于均匀")
                
                if max(feature_max - feature_min, 0) < 0.3:
                    suspicious_features.append("特征动态范围异常窄")
            
            # 图像尺寸
            if analysis['image_info'].get('is_standard_size', False):
                suspicious_features.append("使用标准AI生成尺寸")
            
            # 证件照警告
            if analysis.get('is_id_photo', False):
                suspicious_features = ["证件照可能导致误判"] + suspicious_features
            
            return suspicious_features
            
        except Exception as e:
            logger.warning(f"可疑特征识别失败: {e}")
            return ["特征分析出错: " + str(e)]

# 创建一个全局检测器实例
detector = AIDetector()

def detect_ai_generated(image_path: str) -> Tuple[float, Dict[str, Any]]:
    """
    检测图像是否为AI生成
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        Tuple[float, Dict]: (AI生成概率, 详细分析结果)
    """
    return detector.detect(image_path) 