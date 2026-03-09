import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import pickle
import json
import logging
import matplotlib.pyplot as plt
import time
import shutil
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 模型和数据相关路径
MODEL_DIR = './models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, 'logo_detection_model.h5')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.json')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
FEATURE_VECTORS_PATH = os.path.join(MODEL_DIR, 'feature_vectors.pkl')

class LogoDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        """初始化Logo检测模型"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = None
        self.model = None
        self.history = None
        self.class_indices = None
        self.feature_vectors = {}
        
    def build_model(self, num_classes=None):
        """构建基于ResNet50的模型"""
        if num_classes is not None:
            self.num_classes = num_classes
            
        if self.num_classes is None:
            raise ValueError("必须指定类别数量")
        
        # 加载预训练的ResNet50模型（不包括顶层）
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 冻结基础模型的层
        for layer in self.base_model.layers:
            layer.trainable = False
            
        # 添加自定义顶层
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # 构建完整模型
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"模型构建完成，总共 {self.num_classes} 个类别")
        return self.model
    
    def load_trained_model(self, model_path=None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = TRAINED_MODEL_PATH
            
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
            
        try:
            self.model = load_model(model_path)
            logger.info(f"成功加载模型: {model_path}")
            
            # 加载类别映射
            if os.path.exists(CLASS_MAPPING_PATH):
                with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    self.class_indices = mapping.get('class_indices')
                    self.num_classes = len(self.class_indices)
                    logger.info(f"加载了 {self.num_classes} 个类别的映射")
            else:
                logger.warning("类别映射文件不存在")
                
            # 加载特征向量
            if os.path.exists(FEATURE_VECTORS_PATH):
                with open(FEATURE_VECTORS_PATH, 'rb') as f:
                    self.feature_vectors = pickle.load(f)
                    logger.info(f"加载了 {len(self.feature_vectors)} 个Logo特征向量")
            else:
                logger.warning("特征向量文件不存在")
                
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def prepare_data(self, data_dir, batch_size=32, validation_split=0.2):
        """准备训练和验证数据"""
        # 数据增强配置 - 训练集
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # 数据增强配置 - 验证集（只进行缩放）
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # 训练数据生成器
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # 验证数据生成器
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # 保存类别索引
        self.class_indices = train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        if self.model is None:
            self.build_model(self.num_classes)
            
        logger.info(f"数据准备完成: {len(train_generator.classes)} 个训练样本, " + 
                    f"{len(validation_generator.classes)} 个验证样本, " + 
                    f"{self.num_classes} 个类别")
        
        return train_generator, validation_generator
    
    def unfreeze_layers(self, layers=10):
        """解冻模型的顶层用于微调"""
        if self.model is None:
            logger.error("模型未初始化，无法解冻层")
            return
            
        # 解冻最后几层进行微调
        for layer in self.base_model.layers[-layers:]:
            layer.trainable = True
            
        # 重新编译模型，使用较小的学习率
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"解冻了基础模型的最后 {layers} 层用于微调")
    
    def train(self, train_generator, validation_generator, epochs=20, initial_epochs=10):
        """训练模型，包括初始训练和微调"""
        if self.model is None:
            logger.error("模型未初始化，无法训练")
            return None
            
        # 创建回调函数
        callbacks = [
            ModelCheckpoint(
                TRAINED_MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # 初始训练（冻结的基础模型）
        logger.info(f"开始初始训练 ({initial_epochs} 个周期)...")
        history1 = self.model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # 解冻部分层进行微调
        logger.info("解冻部分层进行微调...")
        self.unfreeze_layers(10)
        
        # 继续训练（微调）
        logger.info(f"继续微调训练 ({epochs-initial_epochs} 个周期)...")
        history2 = self.model.fit(
            train_generator,
            epochs=epochs,
            initial_epoch=initial_epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # 合并训练历史
        self.history = {}
        for key in history1.history:
            self.history[key] = history1.history[key] + history2.history[key]
        
        # 保存训练历史
        with open(HISTORY_PATH, 'w') as f:
            json.dump({k: [float(val) for val in v] for k, v in self.history.items()}, f)
            
        # 保存类别映射
        with open(CLASS_MAPPING_PATH, 'w') as f:
            json.dump({
                'class_indices': self.class_indices,
                'id_to_class': {str(v): k for k, v in self.class_indices.items()}
            }, f)
            
        logger.info(f"模型训练完成，最佳验证准确率: {max(self.history['val_accuracy']):.4f}")
        
        return self.history
    
    def evaluate(self, test_generator):
        """评估模型性能"""
        if self.model is None:
            logger.error("模型未初始化，无法评估")
            return None
            
        # 获取测试结果
        logger.info("开始评估模型...")
        test_loss, test_acc = self.model.evaluate(test_generator)
        logger.info(f"测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")
        
        # 获取类别标签
        class_labels = list(self.class_indices.keys())
        
        # 获取预测结果
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        
        # 获取真实标签
        y_true = test_generator.classes
        
        # 计算并打印分类报告
        report = classification_report(y_true, y_pred, target_names=class_labels)
        logger.info(f"分类报告:\n{report}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'report': report,
            'confusion_matrix': cm
        }
    
    def extract_features(self, image_path):
        """从图像中提取特征向量"""
        if self.model is None:
            logger.error("模型未初始化，无法提取特征")
            return None
            
        try:
            # 加载和预处理图像
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.input_shape[:2])
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 使用模型的中间层提取特征
            feature_extractor = Model(
                inputs=self.model.input,
                outputs=self.model.layers[-3].output  # 使用倒数第三层作为特征
            )
            features = feature_extractor.predict(img_array)
            return features.flatten()
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def extract_dataset_features(self, data_dir):
        """提取整个数据集的特征向量"""
        if self.model is None:
            logger.error("模型未初始化，无法提取特征")
            return False
            
        logger.info(f"开始提取数据集 {data_dir} 的特征向量...")
        
        # 创建特征提取模型
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output  # 使用倒数第三层作为特征
        )
        
        # 清空当前特征向量
        self.feature_vectors = {}
        
        # 遍历所有类别
        for class_name, class_id in self.class_indices.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"类别目录不存在: {class_dir}")
                continue
                
            logger.info(f"处理类别: {class_name}")
            
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.input_shape[:2])
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    features = feature_extractor.predict(img_array, verbose=0)
                    
                    logo_name = f"{class_name}_{img_file}"
                    self.feature_vectors[logo_name] = {
                        'features': features.flatten(),
                        'class': class_name,
                        'path': img_path
                    }
                except Exception as e:
                    logger.error(f"处理图像失败 {img_path}: {e}")
        
        # 保存特征向量
        with open(FEATURE_VECTORS_PATH, 'wb') as f:
            pickle.dump(self.feature_vectors, f)
            
        logger.info(f"成功提取并保存了 {len(self.feature_vectors)} 个特征向量")
        return True
    
    def predict(self, image_path):
        """预测图像的类别"""
        if self.model is None:
            logger.error("模型未初始化，无法预测")
            return None, 0
            
        try:
            # 加载和预处理图像
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.input_shape[:2])
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 预测
            predictions = self.model.predict(img_array)
            
            # 获取最高概率的类别
            predicted_class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_id])
            
            # 获取类别名称
            id_to_class = {v: k for k, v in self.class_indices.items()}
            predicted_class = id_to_class.get(predicted_class_id, f"未知类别_{predicted_class_id}")
            
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, 0
    
    def find_similar_logos(self, image_path, top_k=5):
        """基于特征向量找到相似的Logo"""
        if not self.feature_vectors:
            logger.error("无特征向量数据，无法进行相似度匹配")
            return []
            
        # 提取查询图像的特征
        query_features = self.extract_features(image_path)
        if query_features is None:
            return []
            
        # 计算相似度
        similarities = []
        for name, data in self.feature_vectors.items():
            if isinstance(data, dict) and 'features' in data:
                feature = data['features']
                similarity = np.dot(query_features, feature) / (
                    np.linalg.norm(query_features) * np.linalg.norm(feature)
                )
                similarities.append({
                    'name': name,
                    'similarity': float(similarity),
                    'class': data.get('class'),
                    'path': data.get('path')
                })
            else:
                # 兼容旧版本数据格式
                feature = data
                similarity = np.dot(query_features, feature) / (
                    np.linalg.norm(query_features) * np.linalg.norm(feature)
                )
                similarities.append({
                    'name': name,
                    'similarity': float(similarity)
                })
        
        # 排序并返回前K个
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        if self.history is None:
            logger.error("无训练历史数据，无法绘图")
            return
            
        plt.figure(figsize=(12, 5))
        
        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='训练准确率')
        plt.plot(self.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('周期')
        plt.ylabel('准确率')
        plt.legend()
        
        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
        plt.close()
        
        logger.info(f"训练历史图表已保存到 {os.path.join(MODEL_DIR, 'training_history.png')}")

def prepare_dataset(source_dir, target_dir, min_samples=5, brand_as_class=True):
    """准备数据集，整理结构并确保每个类别有足够的样本
    
    参数:
    source_dir: 源数据目录
    target_dir: 目标数据目录
    min_samples: 每个类别的最小样本数量
    brand_as_class: 如果为True，使用品牌名称作为类别，否则使用顶级目录名称作为类别
    """
    if not os.path.exists(source_dir):
        logger.error(f"源数据集目录不存在: {source_dir}")
        return False
        
    # 创建目标目录
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    logger.info(f"开始准备数据集，从 {source_dir} 到 {target_dir}，品牌作为类别: {brand_as_class}")
    
    # 统计类别和样本数
    class_count = 0
    sample_count = 0
    brand_samples = {}  # 用于统计每个品牌的样本数
    
    # 如果使用品牌作为类别（处理多级目录）
    if brand_as_class:
        # 第一遍：统计每个品牌的样本数
        for category in os.listdir(source_dir):
            category_dir = os.path.join(source_dir, category)
            if not os.path.isdir(category_dir):
                continue
                
            for brand in os.listdir(category_dir):
                brand_dir = os.path.join(category_dir, brand)
                if not os.path.isdir(brand_dir):
                    continue
                    
                images = [f for f in os.listdir(brand_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if brand not in brand_samples:
                    brand_samples[brand] = 0
                    
                brand_samples[brand] += len(images)
        
        # 第二遍：处理符合条件的品牌
        for category in os.listdir(source_dir):
            category_dir = os.path.join(source_dir, category)
            if not os.path.isdir(category_dir):
                continue
                
            for brand in os.listdir(category_dir):
                if brand_samples.get(brand, 0) < min_samples:
                    logger.warning(f"品牌 {brand} 样本数不足 {min_samples}，已跳过")
                    continue
                    
                brand_dir = os.path.join(category_dir, brand)
                if not os.path.isdir(brand_dir):
                    continue
                
                # 创建目标品牌目录
                target_brand_dir = os.path.join(target_dir, brand)
                if not os.path.exists(target_brand_dir):
                    os.makedirs(target_brand_dir)
                    class_count += 1
                
                # 复制图像
                images = [f for f in os.listdir(brand_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img in images:
                    src_path = os.path.join(brand_dir, img)
                    # 使用品牌名和图像名的组合作为新的图像名，避免重名
                    dst_path = os.path.join(target_brand_dir, f"{category}_{img}")
                    shutil.copy2(src_path, dst_path)
                    sample_count += 1
                
                logger.info(f"处理品牌 {brand} (类别 {category}): {len(images)} 个样本")
    else:
        # 原始方法：使用顶级目录作为类别
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # 获取类别下的所有图像
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 跳过样本数不足的类别
            if len(images) < min_samples:
                logger.warning(f"类别 {class_name} 样本数不足 {min_samples}，已跳过")
                continue
                
            # 创建目标类别目录
            target_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(target_class_dir)
            
            # 复制图像
            for img in images:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(target_class_dir, img)
                shutil.copy2(src_path, dst_path)
                sample_count += 1
                
            class_count += 1
            logger.info(f"处理类别 {class_name}: {len(images)} 个样本")
    
    logger.info(f"数据集准备完成: {class_count} 个类别, {sample_count} 个样本")
    return True

def train_logo_model(data_dir, epochs=30, batch_size=32, brand_as_class=True):
    # 训练Logo检测模型的主函数
    logger.info(f"开始训练Logo检测模型，数据目录: {data_dir}")
    
    # 准备临时数据集目录
    temp_data_dir = os.path.join(MODEL_DIR, 'processed_dataset')
    prepare_dataset(data_dir, temp_data_dir, brand_as_class=brand_as_class)
    
    # 创建模型
    model = LogoDetectionModel()
    
    # 准备数据
    train_gen, val_gen = model.prepare_data(temp_data_dir, batch_size=batch_size)
    
    # 训练模型
    initial_epochs = min(10, epochs // 2)
    model.train(train_gen, val_gen, epochs=epochs, initial_epochs=initial_epochs)
    
    # 提取特征向量
    model.extract_dataset_features(temp_data_dir)
    
    # 绘制训练历史
    model.plot_training_history()
    
    logger.info("Logo检测模型训练完成")
    return model

def load_logo_model():
    # 加载已训练的Logo检测模型
    model = LogoDetectionModel()
    if model.load_trained_model():
        return model
    return None

def test_prediction(model, image_path):
    # 测试预测
    if model is None:
        logger.error("模型未加载，无法进行预测")
        return None
        
    # 进行分类预测
    predicted_class, confidence = model.predict(image_path)
    logger.info(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")
    
    # 查找相似Logo
    similar_logos = model.find_similar_logos(image_path, top_k=5)
    logger.info(f"找到 {len(similar_logos)} 个相似Logo:")
    for i, logo in enumerate(similar_logos):
        logger.info(f"  {i+1}. {logo['name']}: {logo['similarity']:.4f}")
    
    return {
        'prediction': {
            'class': predicted_class,
            'confidence': confidence
        },
        'similar_logos': similar_logos
    }

if __name__ == "__main__":
    # train_logo_model('data/train_and_test/train', epochs=20)
    
    # model = load_logo_model()
    # if model:
    #     test_prediction(model, 'test_image.jpg')
    pass 