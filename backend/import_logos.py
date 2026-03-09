import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json
from tqdm import tqdm
import time
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR = '../data/Logo-2K+'

FEATURES_FILE = 'logo_features.pkl'
LOGOS_JSON_FILE = 'logo_data.json'

def extract_features(image_path, model):
    """从图像中提取特征向量"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 使用优化后的模型直接提取特征向量
        features = model.predict(img_array, verbose=0)
        
        print(f"提取的特征向量维度: {features.shape}, 均值: {np.mean(features):.6f}, 标准差: {np.std(features):.6f}")
        
        # 如果不是正好2048维，规范化处理
        if features.shape[1] != 2048:
            print(f"特征向量维度为 {features.shape[1]}，正在调整为 2048")
            if features.shape[1] > 2048:
                features = features[:, :2048]
            else:
                padding = np.zeros((1, 2048 - features.shape[1]))
                features = np.concatenate((features, padding), axis=1)
        
        # 对特征向量进行L2标准化，确保余弦相似度计算正确
        features_norm = np.linalg.norm(features)
        if features_norm > 0:
            normalized_features = features / features_norm
            print(f"特征向量L2标准化后范数: {np.linalg.norm(normalized_features):.6f}")
        else:
            print("警告: 特征向量范数为0，无法标准化")
            normalized_features = features
                
        return normalized_features.flatten()
    except Exception as e:
        print(f"提取特征失败 ({image_path}): {e}")
        print(traceback.format_exc())
        return np.random.rand(2048)

def import_dataset(dataset_path=None):
    """导入Logo数据集并提取特征"""
    try:
        if dataset_path and os.path.exists(dataset_path):
            global DATA_DIR
            DATA_DIR = dataset_path
            print(f"使用指定的数据集路径: {DATA_DIR}")
        else:
            print(f"使用默认数据集路径: {DATA_DIR}")
            
        if not os.path.exists(DATA_DIR):
            print(f"错误: 数据集路径不存在: {DATA_DIR}")
            return {
                'success': False,
                'error': f'数据集路径不存在: {DATA_DIR}'
            }
        

        print("正在加载ResNet50模型...")
        try:
            # 创建基础模型
            base_model_backbone = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model_backbone.trainable = False
            
            # 创建一个完整的特征提取模型，包括全局平均池化层
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model_backbone(inputs)
            outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
            model = tf.keras.Model(inputs, outputs)
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return {
                'success': False,
                'error': f'模型加载失败: {str(e)}'
            }


        logo_features = {}
        logo_data = []


        logo_id = 1
        

        try:
            categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            print(f"找到 {len(categories)} 个分类")
        except Exception as e:
            print(f"读取分类目录失败: {e}")
            return {
                'success': False,
                'error': f'读取分类目录失败: {str(e)}'
            }
        
        if not categories:
            print(f"警告: 未找到任何分类目录")
            return {
                'success': False,
                'error': '未找到任何分类目录'
            }
        
        # 统计数据
        total_logos = 0
        total_brands = 0

        for category in categories:
            category_path = os.path.join(DATA_DIR, category)

            try:
                brands = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
                print(f"在 {category} 分类中找到 {len(brands)} 个品牌")
                total_brands += len(brands)
            except Exception as e:
                print(f"读取 {category} 分类下的品牌失败: {e}")
                continue
            
            if not brands:
                print(f"警告: {category} 分类下未找到任何品牌")
                continue
            

            for brand in tqdm(brands, desc=f"处理 {category} 分类"):
                brand_path = os.path.join(category_path, brand)

                try:
                    images = [f for f in os.listdir(brand_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                except Exception as e:
                    print(f"读取 {brand} 品牌下的图片失败: {e}")
                    continue
                
                if not images:
                    print(f"警告: {brand} 品牌下未找到任何图片")
                    continue

                logo_name = f"{brand} ({category})"
                
                # 初始化这个品牌的特征向量列表
                if logo_name not in logo_features:
                    logo_features[logo_name] = []
                
                # 处理这个品牌的所有图像
                for img_file in images:
                    img_path = os.path.join(brand_path, img_file)
                    try:
                        # 提取特征向量
                        feature_vector = extract_features(img_path, model)
                        
                        # 将特征向量添加到该品牌的列表中
                        logo_features[logo_name].append(feature_vector.tolist())
                        
                        # 添加图像信息到logo_data
                        logo_data.append({
                            'id': logo_id,
                            'name': logo_name,
                            'category': category,
                            'image_path': img_path
                        })
                        
                        logo_id += 1
                        total_logos += 1
                        
                    except Exception as e:
                        print(f"处理图像 {img_path} 失败: {e}")
                        continue
                
                print(f"品牌 {logo_name} 处理了 {len(logo_features[logo_name])} 个Logo图像")
        
        if not logo_features:
            print(f"警告: 未提取到任何Logo特征")
            return {
                'success': False,
                'error': '未提取到任何Logo特征'
            }
        

        try:
            print(f"正在保存 {len(logo_features)} 个品牌的 {total_logos} 个Logo特征到 {FEATURES_FILE}")
            with open(FEATURES_FILE, 'wb') as f:
                pickle.dump(logo_features, f)
        except Exception as e:
            print(f"保存特征向量失败: {e}")
            return {
                'success': False,
                'error': f'保存特征向量失败: {str(e)}'
            }
        

        try:
            print(f"正在保存 {len(logo_data)} 个Logo数据到 {LOGOS_JSON_FILE}")
            with open(LOGOS_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(logo_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存Logo数据失败: {e}")
            return {
                'success': False,
                'error': f'保存Logo数据失败: {str(e)}'
            }
        
        print("导入完成!")
        print(f"总共导入了 {len(logo_features)} 个品牌，{total_logos} 个Logo图像")
        
        return {
            'success': True,
            'logo_count': total_logos,
            'brand_count': len(logo_features),
            'categories': categories,
            'features_file': FEATURES_FILE,
            'logos_file': LOGOS_JSON_FILE
        }
    except Exception as e:
        error_msg = f"导入数据集过程中发生异常: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            'success': False,
            'error': error_msg
        }

def main():

    dataset_path = None
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        

    result = import_dataset(dataset_path)
    

    return 0 if result['success'] else 1

if __name__ == "__main__":
    sys.exit(main()) 