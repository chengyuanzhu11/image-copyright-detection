from flask import Flask, request, jsonify, send_from_directory
import logging
import os
from pathlib import Path
import numpy as np
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import sys
import uuid
import time
import threading
import traceback
import shutil
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='', static_folder='../frontend')

CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     expose_headers=["Content-Type", "Content-Length"]
)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length')
    return response

TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(TEMP_FOLDER):
    try:
        os.makedirs(TEMP_FOLDER)
        logger.info(f"创建临时文件夹: {TEMP_FOLDER}")
    except Exception as e:
        logger.error(f"创建临时文件夹失败: {e}")
        raise

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_DIR = './models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

base_model = None
logo_model = None
model_training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None
}

try:
    base_model_backbone = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model_backbone.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model_backbone(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    base_model = tf.keras.Model(inputs, outputs)
    logger.info("成功加载ResNet50模型")
    
    try:
        from logo_model import load_logo_model
        logo_model = load_logo_model()
        if logo_model:
            logger.info("成功加载Logo检测模型")
        else:
            logger.info("未找到已训练的Logo检测模型，将使用基础特征提取")
    except Exception as e:
        logger.warning(f"加载Logo检测模型失败: {e}")
except Exception as e:
    logger.error(f"加载模型失败: {e}")

LOGOS = []

LOGO_FEATURES = {
    "示例Logo1": [np.random.rand(2048)],
    "示例Logo2": [np.random.rand(2048)],
    "示例Logo3": [np.random.rand(2048)]
}

BRAND_LOGOS = {}  

DETECTION_RESULTS = {}

def load_logo_features():
    features_file = 'logo_features.pkl'
    json_file = 'logo_data.json'
    
    global LOGO_FEATURES, LOGOS, BRAND_LOGOS
    
    try:
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                loaded_features = pickle.load(f)
                
                LOGO_FEATURES = {}
                BRAND_LOGOS = {}
                
                for name, features in loaded_features.items():
                    if isinstance(features, np.ndarray) and features.ndim == 1:
                        features = [features]
                    
                    LOGO_FEATURES[name] = []
                    for feature in features:
                        feature_array = np.array(feature)
                        
                        if len(feature_array) != 2048:
                            logger.warning(f"特征向量 {name} 维度不是2048 ({len(feature_array)}), 进行调整")
                            if len(feature_array) > 2048:
                                feature_array = feature_array[:2048]
                            else:
                                padding = np.zeros(2048 - len(feature_array))
                                feature_array = np.concatenate((feature_array, padding))
                        
                        norm = np.linalg.norm(feature_array)
                        if norm > 0:
                            feature_array = feature_array / norm
                            
                        LOGO_FEATURES[name].append(feature_array)
                
            logger.info(f"从 {features_file} 加载了 {len(LOGO_FEATURES)} 个品牌特征")
        else:
            logger.warning(f"特征文件 {features_file} 不存在，使用示例特征")
            # 只生成几个示例特征，用于测试
            LOGO_FEATURES = {
                "示例Logo1": [np.random.rand(2048)],
                "示例Logo2": [np.random.rand(2048)],
                "示例Logo3": [np.random.rand(2048)]
            }
            
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                LOGOS = json.load(f)
                
                # 构建品牌到Logo文件的映射
                for logo in LOGOS:
                    brand_name = logo.get('name')
                    if brand_name not in BRAND_LOGOS:
                        BRAND_LOGOS[brand_name] = []
                    
                    # 添加这个logo到对应品牌的列表中
                    BRAND_LOGOS[brand_name].append(logo)
                
            logger.info(f"从 {json_file} 加载了 {len(LOGOS)} 个Logo数据，共 {len(BRAND_LOGOS)} 个品牌")
        else:
            logger.warning(f"Logo数据文件 {json_file} 不存在，使用空列表")
            LOGOS = []
            BRAND_LOGOS = {}
            
    except Exception as e:
        logger.error(f"加载Logo特征时出错: {e}")
        logger.error(traceback.format_exc())

load_logo_features()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """从图像中提取特征向量"""
    try:
        if base_model is None:
            logger.error("基础模型未加载，无法提取特征")
            return np.random.rand(2048)
            
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 使用优化后的模型直接提取特征向量
        features = base_model.predict(img_array)
        
        # 记录特征向量的基本信息，用于调试
        logger.info(f"提取的特征向量维度: {features.shape}, 均值: {np.mean(features):.6f}, 标准差: {np.std(features):.6f}")
        
        # 如果不是正好2048维，规范化处理
        if features.shape[1] != 2048:
            logger.warning(f"特征向量维度为 {features.shape[1]}，正在调整为 2048")
            if features.shape[1] > 2048:
                features = features[:, :2048]
            else:
                padding = np.zeros((1, 2048 - features.shape[1]))
                features = np.concatenate((features, padding), axis=1)
        
        features_norm = np.linalg.norm(features)
        if features_norm > 0:
            normalized_features = features / features_norm
            logger.info(f"特征向量L2标准化后范数: {np.linalg.norm(normalized_features):.6f}")
        else:
            logger.warning("特征向量范数为0，无法标准化")
            normalized_features = features
                
        return normalized_features.flatten()
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        logger.error(traceback.format_exc())
        return np.random.rand(2048)

def calculate_similarity(query_features, logo_features):
    try:
        query_vec = np.array(query_features)
        logo_vec = np.array(logo_features)
        
        if not (np.all(np.isfinite(query_vec)) and np.all(np.isfinite(logo_vec))):
            logger.error("特征向量包含非有限值 (NaN或Inf)")
            return 0.0
            
        query_norm = np.linalg.norm(query_vec)
        logo_norm = np.linalg.norm(logo_vec)
        
        if query_norm == 0 or logo_norm == 0:
            logger.warning("特征向量范数为0，无法计算相似度")
            return 0.0
            
        if not np.isclose(query_norm, 1.0, atol=1e-3):
            query_vec = query_vec / query_norm
            
        if not np.isclose(logo_norm, 1.0, atol=1e-3):
            logo_vec = logo_vec / logo_norm
            
        similarity = np.dot(query_vec, logo_vec)
        
        if not (-1.0 <= similarity <= 1.0):
            logger.warning(f"计算出异常相似度值: {similarity}, 已调整为有效范围")
            similarity = max(-1.0, min(1.0, similarity))
            
        return float(similarity)
    except Exception as e:
        logger.error(f"计算相似度时出错: {e}")
        logger.error(traceback.format_exc())
        return 0.0

def calculate_brand_similarity(query_features, brand_features_list):
    try:
        if not isinstance(brand_features_list, list):
            logger.warning("品牌特征不是列表格式，转换为单元素列表")
            brand_features_list = [brand_features_list]
        
        max_similarity = 0.0
        similarities = []
        
        for logo_features in brand_features_list:
            similarity = calculate_similarity(query_features, logo_features)
            similarities.append(similarity)
        
        if similarities:
            max_similarity = max(similarities)
            logger.info(f"计算了{len(similarities)}个logo特征相似度，最大值: {max_similarity:.4f}")
        
        return max_similarity
    except Exception as e:
        logger.error(f"计算品牌相似度时出错: {e}")
        logger.error(traceback.format_exc())
        return 0.0

def detect_logo_similarity(image_path):
    """检测图像与数据库中Logo的相似度"""
    try:
        # 首先尝试使用训练好的模型进行预测
        if logo_model is not None:
            try:
                predicted_class, confidence = logo_model.predict(image_path)
                if predicted_class and confidence > 0.7:
                    return f"检测到Logo: {predicted_class}, 置信度: {confidence:.2f}"
            except Exception as e:
                logger.warning(f"使用训练模型预测失败: {e}")
        
        # 回退到特征相似度方法
        query_features = extract_features(image_path)
        
        # 使用更低的阈值以增加检测灵敏度
        similarity_threshold = 0.6
        
        # 找出所有相似度高于阈值的Logo
        similar_logos = []
        
        # 保存相似度分布统计信息
        similarities = []
        
        for logo_name, features in LOGO_FEATURES.items():
            try:
                # 确保特征向量形状一致
                if len(features) != len(query_features):
                    logger.warning(f"特征向量维度不匹配: {logo_name} ({len(features)}) vs 查询图片 ({len(query_features)})")
                    continue
                
                # 确保特征向量已经标准化
                logo_features = np.array(features)
                logo_norm = np.linalg.norm(logo_features)
                if logo_norm > 0 and not np.isclose(logo_norm, 1.0, atol=1e-3):
                    logger.warning(f"Logo '{logo_name}' 特征向量未标准化 (范数={logo_norm:.4f})，正在标准化")
                    logo_features = logo_features / logo_norm
                    
                # 计算余弦相似度    
                similarity = calculate_similarity(query_features, logo_features)
                similarities.append(similarity)
                
                logo_category = None
                if LOGOS:
                    for logo in LOGOS:
                        if logo.get('name') == logo_name:
                            logo_category = logo.get('category')
                            break
                
                similar_logos.append({
                    'name': logo_name,
                    'category': logo_category,
                    'similarity': float(similarity)
                })
            except Exception as e:
                logger.error(f"计算与 {logo_name} 的相似度失败: {e}")
                logger.error(traceback.format_exc())
        
        # 记录相似度统计信息
        if similarities:
            logger.info(f"相似度统计: 最小={min(similarities):.4f}, 最大={max(similarities):.4f}, 平均={np.mean(similarities):.4f}, 中位数={np.median(similarities):.4f}")
        
        # 按相似度排序
        similar_logos.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 获取最相似的Logo
        max_similarity = similar_logos[0]['similarity'] if similar_logos else 0
        similar_logo = similar_logos[0]['name'] if similar_logos else None
        logo_category = similar_logos[0]['category'] if similar_logos else None
        
        logger.info(f"最高相似度: {max_similarity:.4f}, Logo: {similar_logo}")
        
        if max_similarity > similarity_threshold:
            result = f"检测到相似Logo: {similar_logo}"
            if logo_category:
                result += f", 分类: {logo_category}"
            result += f", 相似度: {max_similarity:.2f}"
            return result
        else:
            return f"未检测到相似Logo (相似度阈值: {similarity_threshold})"
            
    except Exception as e:
        logger.error(f"Logo检测过程中出错: {e}")
        logger.error(traceback.format_exc())
        return f"检测过程出错: {str(e)}"

@app.route('/detect', methods=['POST'])
def detect_logo():
    """处理图片版权检测请求"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传图片文件'}), 400
                
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(image.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        temp_path = os.path.join(TEMP_FOLDER, image.filename)
        image.save(temp_path)
        
        detection_id = str(uuid.uuid4())
        
        # 尝试使用训练好的模型进行预测
        model_prediction = None
        if logo_model is not None:
            try:
                predicted_class, confidence = logo_model.predict(temp_path)
                if predicted_class:
                    model_prediction = {
                        'class': predicted_class,
                        'confidence': float(confidence)
                    }
                    
                    # 如果有足够的置信度，直接使用模型预测结果
                    if confidence > 0.7:
                        result_text = f"检测到Logo: {predicted_class}, 置信度: {confidence:.2f}"
                        detection_method = "trained_model"
                        
                        DETECTION_RESULTS[detection_id] = {
                            'timestamp': time.time(),
                            'filename': image.filename,
                            'file_path': temp_path,
                            'main_result': result_text,
                            'model_prediction': model_prediction,
                            'detection_method': detection_method
                        }
                        
                        # 异步获取相似度匹配结果并更新
                        def update_with_similarities():
                            try:
                                # 获取模型的相似度匹配结果
                                similar_logos = logo_model.find_similar_logos(temp_path, top_k=10)
                                
                                if detection_id in DETECTION_RESULTS:
                                    DETECTION_RESULTS[detection_id]['similar_logos'] = similar_logos
                            except Exception as e:
                                logger.error(f"获取相似Logo失败: {e}")
                                
                        thread = threading.Thread(target=update_with_similarities)
                        thread.daemon = True
                        thread.start()
                        
                        return jsonify({
                            'result': result_text,
                            'detection_id': detection_id,
                            'method': 'trained_model'
                        })
            except Exception as e:
                logger.warning(f"模型预测失败: {e}")
        
        # 如果模型预测不可用或置信度不够，回退到特征相似度方法
        query_features = extract_features(temp_path)
        
        # 使用更低的阈值以增加检测灵敏度
        similarity_threshold = 0.6
        
        # 找出所有相似度高于阈值的Logo
        similar_logos = []
        
        # 保存相似度分布统计信息
        similarities = []
        
        for logo_name, features_list in LOGO_FEATURES.items():
            try:
                # 确保特征向量形状一致 - 现在循环中的features_list是一个列表
                if not features_list:
                    logger.warning(f"品牌 {logo_name} 没有有效的特征向量")
                    continue
                
                # 计算与该品牌所有logo的最大相似度
                similarity = calculate_brand_similarity(query_features, features_list)
                similarities.append(similarity)
                    
                logo_category = None
                
                if LOGOS:
                    for logo in LOGOS:
                        if logo.get('name') == logo_name:
                            logo_category = logo.get('category')
                            break
                
                similar_logos.append({
                    'name': logo_name,
                    'category': logo_category,
                    'similarity': float(similarity),
                    'logo_count': len(features_list)  # 添加该品牌的logo数量信息
                })
            except Exception as e:
                logger.error(f"计算与 {logo_name} 的相似度失败: {e}")
                logger.error(traceback.format_exc())
        
        # 记录相似度统计信息
        if similarities:
            logger.info(f"相似度统计: 最小={min(similarities):.4f}, 最大={max(similarities):.4f}, 平均={np.mean(similarities):.4f}, 中位数={np.median(similarities):.4f}")
        
        # 按相似度排序
        similar_logos.sort(key=lambda x: x['similarity'], reverse=True)
        
        most_similar = similar_logos[0] if similar_logos else None
        
        logger.info(f"最高相似度: {most_similar['similarity'] if most_similar else 0:.4f}")
        
        if most_similar and most_similar['similarity'] > similarity_threshold:
            result_text = f"检测到相似Logo: {most_similar['name']}"
            if most_similar['category']:
                result_text += f", 分类: {most_similar['category']}"
            result_text += f", 相似度: {most_similar['similarity']:.2f}"
            detection_method = "feature_similarity"
        else:
            result_text = f"未检测到相似Logo (相似度阈值: {similarity_threshold})"
            detection_method = "none"
        
        DETECTION_RESULTS[detection_id] = {
            'timestamp': time.time(),
            'filename': image.filename,
            'file_path': temp_path,
            'main_result': result_text,
            'model_prediction': model_prediction,
            'matches': similar_logos[:10],
            'detection_method': detection_method
        }
        
        return jsonify({
            'result': result_text,
            'detection_id': detection_id,
            'method': detection_method
        })
            
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/detailed_result', methods=['GET'])
def get_detailed_result():
    """获取详细的检测结果"""
    try:
        detection_id = request.args.get('id')
        if not detection_id or detection_id not in DETECTION_RESULTS:
            return jsonify({'error': '未找到检测结果或结果已过期'}), 404
            
        result = DETECTION_RESULTS[detection_id]
        
        current_time = time.time()
        to_delete = []
        for key, value in DETECTION_RESULTS.items():
            if current_time - value.get('timestamp', 0) > 3600:  
                to_delete.append(key)
        
        for key in to_delete:
            if key in DETECTION_RESULTS:
                del DETECTION_RESULTS[key]
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching detailed result: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_logo', methods=['POST'])
def add_logo():
    """添加新Logo到内存数据中"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传图片文件'}), 400
                
        image = request.files['image']
        logo_name = request.form.get('name', '')
        
        if image.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not logo_name:
            return jsonify({'error': 'Logo name is required'}), 400
            
        if not allowed_file(image.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        temp_path = os.path.join(TEMP_FOLDER, image.filename)
        image.save(temp_path)
        
        feature_vector = extract_features(temp_path)
        
        new_logo = {
            'id': len(LOGOS) + 1, 
            'name': logo_name, 
            'feature_vector': feature_vector
        }
        LOGOS.append(new_logo)
        
        return jsonify({'success': f'Logo {logo_name} 添加成功'})
            
    except Exception as e:
        logger.error(f"Error adding logo: {e}")
        return jsonify({'error': f'Failed to add logo: {str(e)}'}), 500

@app.route('/logos', methods=['GET'])
def get_logos():
    """获取所有Logo列表"""
    try:
        category = request.args.get('category')
        
        if category:
            if LOGOS:
                logo_list = [{'id': logo['id'], 'name': logo['name'], 'category': logo.get('category')} 
                            for logo in LOGOS if logo.get('category') == category]
            else:
                logo_list = [{'id': idx+1, 'name': name, 'category': category} 
                           for idx, name in enumerate(LOGO_FEATURES.keys()) 
                           if name.endswith(f"({category})")]
        else:
            if LOGOS:
                logo_list = [{'id': logo['id'], 'name': logo['name'], 'category': logo.get('category')} 
                           for logo in LOGOS]
            else:
                logo_list = [{'id': idx+1, 'name': name} for idx, name in enumerate(LOGO_FEATURES.keys())]
                
        return jsonify(logo_list)
    except Exception as e:
        logger.error(f"Error fetching logos: {e}")
        
        return jsonify({'error': str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """获取所有Logo分类"""
    try:
        if LOGOS:
            categories = list(set(logo.get('category') for logo in LOGOS if logo.get('category')))
        else:
            categories = []
            for name in LOGO_FEATURES.keys():
                if '(' in name and ')' in name:
                    category = name.split('(')[-1].strip(')')
                    if category and category not in categories:
                        categories.append(category)
        
        categories.sort()  
        return jsonify(categories)
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model_api():
    """训练Logo检测模型"""
    global model_training_status, logo_model
    
    if not request.is_json:
        return jsonify({'error': 'Expected JSON data'}), 400
        
    # 如果已经在训练中
    if model_training_status['is_training']:
        return jsonify({
            'status': 'training',
            'message': model_training_status['message'],
            'progress': model_training_status['progress']
        })
        
    data = request.json
    dataset_path = data.get('dataset_path', 'data/train_and_test/train')
    epochs = data.get('epochs', 30)
    batch_size = data.get('batch_size', 32)
    brand_as_class = data.get('brand_as_class', True) 
    
    # 检查数据集路径
    if not os.path.exists(dataset_path):
        return jsonify({'error': f'数据集路径不存在: {dataset_path}'}), 404
        
    # 更新训练状态
    model_training_status = {
        'is_training': True,
        'progress': 0,
        'message': '正在准备训练数据...',
        'start_time': time.time(),
        'end_time': None
    }
    
    def train_model_thread():
        """在后台线程中训练模型"""
        global model_training_status, logo_model
        
        try:
            from logo_model import train_logo_model
            
            # 更新进度
            model_training_status['message'] = '正在训练模型...'
            model_training_status['progress'] = 10
            
            # 训练模型
            logo_model = train_logo_model(
                dataset_path, 
                epochs, 
                batch_size, 
                brand_as_class=brand_as_class
            )
            
            # 训练完成
            model_training_status['is_training'] = False
            model_training_status['progress'] = 100
            model_training_status['message'] = '模型训练完成'
            model_training_status['end_time'] = time.time()
            
            logger.info("Logo检测模型训练完成")
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            model_training_status['is_training'] = False
            model_training_status['message'] = f'训练失败: {str(e)}'
            model_training_status['end_time'] = time.time()
    
    # 启动训练线程
    thread = threading.Thread(target=train_model_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': '模型训练已启动',
        'dataset_path': dataset_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'brand_as_class': brand_as_class
    })

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """获取模型训练状态"""
    global model_training_status, logo_model
    
    # 检查Logo检测模型
    has_trained_model = logo_model is not None
    
    response = {
        'training_status': model_training_status,
        'has_trained_model': has_trained_model,
        'base_model_loaded': base_model is not None
    }
    
    # 如果有训练好的模型，添加其信息
    if has_trained_model:
        # 尝试获取模型信息
        try:
            class_count = len(logo_model.class_indices) if hasattr(logo_model, 'class_indices') else 0
            feature_count = len(logo_model.feature_vectors) if hasattr(logo_model, 'feature_vectors') else 0
            
            response['model_info'] = {
                'class_count': class_count,
                'feature_count': feature_count,
                'classes': list(logo_model.class_indices.keys()) if hasattr(logo_model, 'class_indices') else []
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            response['model_info'] = {
                'error': str(e)
            }
    
    return jsonify(response)

@app.route('/import_dataset', methods=['POST'])
def import_dataset_api():
    """导入Logo数据集并提取特征"""
    if not request.is_json:
        return jsonify({'error': 'Expected JSON data'}), 400
        
    data = request.json
    dataset_path = data.get('dataset_path')
    regenerate_features = data.get('regenerate_features', False)
    
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': f'数据集路径不存在或无效: {dataset_path}'}), 404
        
    # 创建并启动导入线程
    def run_import():
        try:
            from import_logos import import_dataset
            result = import_dataset(dataset_path)
            
            if result.get('success'):
                # 导入成功后重新加载特征和数据
                global LOGO_FEATURES, LOGOS, BRAND_LOGOS
                if regenerate_features:
                    # 使用新的特征提取方法重新生成所有特征
                    logger.info("开始重新生成所有Logo特征...")
                    
                    # 重新加载所有Logo数据
                    with open('logo_data.json', 'r', encoding='utf-8') as f:
                        LOGOS = json.load(f)
                    
                    # 清空特征存储
                    LOGO_FEATURES = {}
                    BRAND_LOGOS = {}
                    
                    # 对每个Logo重新提取特征
                    for logo in LOGOS:
                        image_path = logo.get('image_path')
                        if image_path and os.path.exists(image_path):
                            try:
                                feature = extract_features(image_path)
                                LOGO_FEATURES[logo.get('name')] = [feature]
                                logger.info(f"重新生成特征: {logo.get('name')}")
                            except Exception as e:
                                logger.error(f"重新生成特征失败 ({logo.get('name')}): {e}")
                    
                    # 保存新的特征文件
                    with open('logo_features.pkl', 'wb') as f:
                        pickle.dump(LOGO_FEATURES, f)
                    
                    logger.info(f"重新生成了 {len(LOGO_FEATURES)} 个Logo特征并保存")
                else:
                    # 简单重新加载数据
                    load_logo_features()
                    
                logger.info("数据集导入和特征加载完成")
            else:
                logger.error(f"数据集导入失败: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"导入过程中出错: {e}")
            logger.error(traceback.format_exc())
    
    thread = threading.Thread(target=run_import)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'正在导入数据集: {dataset_path}',
        'regenerate_features': regenerate_features
    })

@app.route('/reset_features', methods=['POST'])
def reset_features():
    try:
        logger.info("开始重置特征向量库...")
        
        if not os.path.exists('logo_data.json'):
            return jsonify({
                'success': False,
                'error': '找不到Logo数据文件 logo_data.json'
            }), 404
        
        with open('logo_data.json', 'r', encoding='utf-8') as f:
            LOGOS = json.load(f)
        
        logger.info(f"加载了 {len(LOGOS)} 个Logo数据记录")
        
        brand_logos = {}
        for logo in LOGOS:
            brand_name = logo.get('name')
            if brand_name not in brand_logos:
                brand_logos[brand_name] = []
            brand_logos[brand_name].append(logo)
        
        logger.info(f"识别出 {len(brand_logos)} 个不同品牌")
        
        logo_features = {}
        processed_count = 0
        error_count = 0
        
        if os.path.exists('logo_features.pkl'):
            backup_file = f'logo_features_{int(time.time())}.pkl.bak'
            try:
                shutil.copy('logo_features.pkl', backup_file)
                logger.info(f"已创建特征文件备份: {backup_file}")
            except Exception as e:
                logger.error(f"创建备份失败: {e}")
        
        for brand_name, logos in brand_logos.items():
            logo_features[brand_name] = []
            
            # 记录处理进度
            logger.info(f"处理品牌 {brand_name} 的 {len(logos)} 个Logo图像")
            
            for logo in logos:
                try:
                    image_path = logo.get('image_path')
                    
                    if not image_path or not os.path.exists(image_path):
                        logger.warning(f"图像不存在: {image_path}")
                        error_count += 1
                        continue
                        
                    feature = extract_features(image_path)
                    logo_features[brand_name].append(feature)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"处理 {brand_name} 的Logo时出错: {e}")
                    logger.error(traceback.format_exc())
                    error_count += 1
        
        # 保存新的特征文件
        with open('logo_features.pkl', 'wb') as f:
            pickle.dump(logo_features, f)
        
        # 重新加载特征
        global LOGO_FEATURES, BRAND_LOGOS
        LOGO_FEATURES = {}
        BRAND_LOGOS = {}
        load_logo_features()
        
        return jsonify({
            'success': True,
            'message': f'成功重置特征库，处理了 {processed_count} 个Logo，失败 {error_count} 个，共 {len(logo_features)} 个品牌',
            'processed': processed_count,
            'errors': error_count,
            'total_brands': len(logo_features),
            'total_features': sum(len(features) for features in LOGO_FEATURES.values())
        })
    except Exception as e:
        logger.error(f"重置特征库时出错: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({'error': 'Resource not found'}), 404

@app.route('/detect_ai_generated', methods=['POST'])
def detect_ai_generated():
    """检测图片是否为AI生成"""
    try:
        # 检查是否允许客户端验证
        client_validation = request.args.get('client_validation', 'false') == 'true'
        
        if 'image' not in request.files:
            logger.warning("未上传图片文件")
            return jsonify({'error': '未上传图片文件'}), 400
                
        image = request.files['image']
        if image.filename == '':
            logger.warning("空文件名")
            return jsonify({'error': '未选择任何文件'}), 400
        
        if not allowed_file(image.filename):
            allowed_extensions = ', '.join(ALLOWED_EXTENSIONS)
            logger.warning(f"不支持的文件类型: {image.filename}")
            return jsonify({'error': f'不支持的文件类型。请上传 {allowed_extensions} 格式的图片'}), 400

        # 生成唯一的临时文件名
        temp_filename = f"ai_check_{int(time.time())}_{secure_filename(image.filename)}"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        try:
            # 保存上传的图片
            image.save(temp_path)
            logger.info(f"图片已保存到: {temp_path}")
            
            # 验证文件是否成功保存及其大小
            if not os.path.exists(temp_path):
                raise Exception("文件保存失败")
            
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                raise Exception("上传的文件是空的")
            
            # 获取前端传入的阈值参数
            threshold = request.form.get('threshold', '50')
            try:
                threshold_value = float(threshold) / 100
                os.environ['AI_DETECTION_THRESHOLD'] = str(threshold_value)
                logger.info(f"使用阈值: {threshold}%")
            except (ValueError, TypeError):
                logger.warning(f"无效的阈值参数: {threshold}，使用默认阈值")
                os.environ['AI_DETECTION_THRESHOLD'] = "0.5"
                threshold_value = 0.5
            
            # 获取检测模式
            detection_mode = request.form.get('mode', 'standard')
            if detection_mode == 'sensitive':
                os.environ['AI_DETECTION_THRESHOLD'] = "0.3"
                threshold_value = 0.3
            elif detection_mode == 'balanced':
                os.environ['AI_DETECTION_THRESHOLD'] = "0.5"
                threshold_value = 0.5
            elif detection_mode == 'strict': 
                os.environ['AI_DETECTION_THRESHOLD'] = "0.7"
                threshold_value = 0.7
            
            # 记录用户信息
            user_id = request.form.get('user_id', 'anonymous')
            detection_id = str(uuid.uuid4())
            
            try:
                # 使用新的AI检测模块
                try:
                    from ai_detector import detect_ai_generated as ai_detect
                    
                    # 执行检测
                    logger.info(f"开始AI检测: {temp_path}")
                    score, analysis = ai_detect(temp_path)
                    logger.info(f"检测完成，得分: {score}")
                    
                except ImportError as e:
                    # 如果导入失败，使用基本图像分析
                    logger.warning(f"AI检测模块导入失败，使用基本分析: {e}")
                    score, analysis = fallback_detection(temp_path)
                    
                # 处理结果 - 确保转换为普通Python类型避免序列化问题
                ai_score = float(score)
                
                # 判断是否是证件照，已在AI检测模块中处理
                is_id_photo = analysis.get('is_id_photo', False)
                if is_id_photo:
                    logger.info(f"检测到可能的证件照，分数: {ai_score}")
                
                # 使用AI检测模块中的结论
                conclusion = analysis.get('conclusion', "AI生成" if ai_score > threshold_value else "真实照片")
                
                # 获取可疑特征
                suspicious_features = analysis.get('suspicious_features', [])
                
                # 构建返回结果
                result = {
                    'detection_id': detection_id,
                    'ai_generated_probability': ai_score,
                    'conclusion': conclusion,
                    'analysis': analysis,
                    'suspicious_features': suspicious_features,
                    'timestamp': time.time(),
                    'threshold': threshold_value * 100,
                    'user_id': user_id,
                    'detector_type': analysis.get('detector_type', 'basic_detector')
                }
                
                # 存储结果
                DETECTION_RESULTS[detection_id] = result
                
                # 更新用户统计
                try:
                    updateUserStats(user_id, "ai_detection", ai_score > threshold_value)
                except Exception as e:
                    logger.warning(f"更新用户统计失败: {e}")
                
                # 使用convert_numpy_types函数处理结果，确保没有NumPy类型
                result = convert_numpy_types(result)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"AI检测处理失败: {e}")
                logger.error(traceback.format_exc())
                # 返回基本错误响应而不是500
                error_result = {
                    'ai_generated_probability': 0.5, 
                    'conclusion': '无法确定',
                    'error_details': str(e),
                    'detection_id': detection_id,
                    'timestamp': time.time()
                }
                return jsonify(convert_numpy_types(error_result))
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"临时文件已删除: {temp_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
            
    except Exception as e:
        logger.error(f"AI检测请求处理失败: {e}")
        return jsonify({'error': str(e)}), 500

def fallback_detection(image_path):
    """当AI模型不可用时的降级检测方法"""
    logger.info(f"使用降级检测模式: {image_path}")
    try:
        # 使用PIL读取图像信息
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
        mode = img.mode
        format_info = getattr(img, 'format', 'Unknown')
        
        # 简单分析图像属性
        is_large = (width * height) > (1024 * 1024)
        is_small = (width * height) < (256 * 256)
        has_alpha = 'A' in mode
        is_png = format_info == 'PNG'
        is_jpeg = format_info in ('JPEG', 'JPG')
        
        # 基于简单规则的得分
        # AI生成图像通常分辨率适中，可能是PNG，可能有透明通道
        score = 0.5  # 默认得分
        
        if is_png:
            score += 0.1  # PNG更常见于AI生成
        if has_alpha:
            score += 0.05  # 透明通道更常见于AI生成
        if is_large:
            score -= 0.05  # 超大图像不太像AI生成
        if is_small:
            score -= 0.1  # 太小的图像也不太像AI生成
        
        # 构建简单分析结果
        analysis = {
            'ai_generated_probability': float(score),
            'is_id_photo': False,
            'using_fallback': True,
            'image_info': {
                'width': width,
                'height': height,
                'format': format_info,
                'mode': mode,
                'has_alpha': has_alpha
            },
            'conclusion': 'AI生成' if score > 0.5 else '真实照片',
            'confidence': 'low',  # 标记置信度低
            'suspicious_features': [],
            'detector_type': 'fallback_basic'
        }
        
        return float(score), analysis
    except Exception as e:
        logger.error(f"降级检测也失败: {e}")
        # 返回默认分析结果
        return 0.5, {
            'ai_generated_probability': 0.5,
            'conclusion': '无法确定',
            'error': '降级检测方法失败',
            'using_fallback': True,
            'detector_type': 'error_fallback'
        }

def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，解决JSON序列化问题"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_suspicious_features(analysis):
    """提取可疑特征"""
    suspicious_features = []
    
    # 检查噪声分析结果
    if "noise_analysis" in analysis:
        noise_std = analysis["noise_analysis"].get("noise_std", 0)
        if noise_std < 10:  # 极低噪声标准差通常表示AI生成
            suspicious_features.append("噪声分布异常均匀")
    
    # 检查纹理分析结果
    if "texture_analysis" in analysis:
        texture_var = analysis["texture_analysis"].get("texture_var", 0)
        if texture_var < 1000:  # 低纹理方差通常表示AI生成
            suspicious_features.append("纹理特征不自然")
    
    # 检查特征统计
    if "feature_statistics" in analysis:
        feature_max = analysis["feature_statistics"].get("max", 0)
        feature_min = analysis["feature_statistics"].get("min", 0)
        feature_std = analysis["feature_statistics"].get("std", 1)
        
        if max(feature_max - feature_min, 0) < 0.3:
            suspicious_features.append("特征动态范围异常窄")
        
        if feature_std < 0.15:
            suspicious_features.append("特征分布过于均匀")
    
    # 如果是证件照，添加提示
    if analysis.get('is_id_photo', False):
        suspicious_features = ["证件照可能导致误判"] + suspicious_features
        
    return suspicious_features

# 新增：更新用户统计数据
def updateUserStats(user_id, detection_type, is_risky):
    """更新用户的统计数据"""
    if not user_id or user_id == 'anonymous':
        return
        
    stats_file = f"user_stats_{user_id}.json"
    stats = {}
    
    # 尝试加载现有统计数据
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        except:
            pass
    
    # 确保存在基本结构
    if 'uploads' not in stats:
        stats['uploads'] = 0
    if 'ai_detected' not in stats:
        stats['ai_detected'] = 0
    if 'logo_detected' not in stats:
        stats['logo_detected'] = 0
    if 'safe_images' not in stats:
        stats['safe_images'] = 0
        
    # 更新计数
    stats['uploads'] += 1
    
    if detection_type == 'ai_detection' and is_risky:
        stats['ai_detected'] += 1
    elif detection_type == 'logo_detection' and is_risky:
        stats['logo_detected'] += 1
    else:
        stats['safe_images'] += 1
    
    # 保存更新后的统计
    try:
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"保存用户统计失败: {e}")

@app.route('/delete_logo/<logo_id>', methods=['DELETE'])
def delete_logo(logo_id):
    try:
        # 查找要删除的Logo
        logo_to_delete = None
        for logo in LOGOS:
            if logo.get('id') == logo_id:
                logo_to_delete = logo
                break
        
        if not logo_to_delete:
            return jsonify({'error': 'Logo不存在'}), 404
            
        # 删除Logo文件
        logo_path = logo_to_delete.get('path')
        if logo_path and os.path.exists(logo_path):
            os.remove(logo_path)
            
        # 从特征中删除
        if logo_to_delete.get('name') in LOGO_FEATURES:
            del LOGO_FEATURES[logo_to_delete.get('name')]
            
        # 从品牌映射中删除
        brand_name = logo_to_delete.get('name')
        if brand_name in BRAND_LOGOS:
            BRAND_LOGOS[brand_name] = [logo for logo in BRAND_LOGOS[brand_name] if logo.get('id') != logo_id]
            if not BRAND_LOGOS[brand_name]:
                del BRAND_LOGOS[brand_name]
                
        # 从Logo列表中删除
        LOGOS[:] = [logo for logo in LOGOS if logo.get('id') != logo_id]
        
        # 保存更新后的数据
        with open('logo_data.json', 'w', encoding='utf-8') as f:
            json.dump(LOGOS, f, ensure_ascii=False, indent=2)
            
        with open('logo_features.pkl', 'wb') as f:
            pickle.dump(LOGO_FEATURES, f)
            
        return jsonify({'message': 'Logo删除成功'})
        
    except Exception as e:
        logger.error(f"删除Logo时出错: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '删除Logo失败'}), 500

if __name__ == '__main__':
    try:
        logger.info("正在启动服务器...")
        logger.info(f"静态文件目录: {app.static_folder}")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        logger.error(traceback.format_exc()) 