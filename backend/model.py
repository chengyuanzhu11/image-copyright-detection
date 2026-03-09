import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

LOGO_DB_PATH = Path("logo_features.pkl")
if LOGO_DB_PATH.exists():
    import pickle
    with open(LOGO_DB_PATH, 'rb') as f:
        loaded_features = pickle.load(f)
    # pkl文件中每个品牌的特征可能是列表格式，取第一个特征向量
    LOGO_FEATURES = {}
    for name, features in loaded_features.items():
        if isinstance(features, list) and len(features) > 0:
            LOGO_FEATURES[name] = np.array(features[0]).flatten()
        elif isinstance(features, np.ndarray):
            LOGO_FEATURES[name] = features.flatten()
    logger.info(f"从 {LOGO_DB_PATH} 加载了 {len(LOGO_FEATURES)} 个Logo特征")
else:
    LOGO_FEATURES = {}

if not LOGO_FEATURES:
    LOGO_FEATURES = {
        "Logo1": np.random.rand(2048),
        "Logo2": np.random.rand(2048),
        "Logo3": np.random.rand(2048)
    }
    logger.info("使用模拟Logo数据进行测试")

datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)


# train_generator = datagen.flow_from_directory('data/Logo-2K+/Logo-2K+', target_size=(224, 224), batch_size=32, class_mode='categorical')

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

def create_model(num_classes):
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# model.fit(train_generator, epochs=10)

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    return image_array

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    features = base_model.predict(img_array)
    # 对空间维度做全局平均池化: (1, 7, 7, 2048) -> (1, 2048)
    if len(features.shape) == 4:
        features = np.mean(features, axis=(1, 2))
    return features.flatten()

def detect_logo_similarity(image_path):
    """
    检测图像与数据库中Logo的相似度
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        query_features = extract_features(image_path)
        
        max_similarity = 0
        similar_logo = None
        
        for logo_name, features in LOGO_FEATURES.items():
            similarity = cosine_similarity([query_features], [features])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                similar_logo = logo_name
        
        try:
            from app import Logo, db
            
            logos = Logo.query.all()
            
            for logo in logos:
                similarity = cosine_similarity([query_features], [logo.feature_vector])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_logo = logo.name
        except Exception as e:
            logger.warning(f"从数据库查询Logo时出错: {str(e)}")
        
        if max_similarity > 0.8:  
            return f"检测到相似Logo: {similar_logo}, 相似度: {max_similarity:.2f}"
        else:
            return "未检测到相似Logo"
    except Exception as e:
        logger.error(f"Logo检测过程中出错: {str(e)}")
        return f"检测过程出错: {str(e)}"

def save_logo_feature(image, logo_name):
    features = extract_features(image)
    LOGO_FEATURES[logo_name] = features
    joblib.dump(LOGO_FEATURES, LOGO_DB_PATH)
    return True

def process_logo_dataset(dataset_path, db):
    from app import Logo, db
    
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            class_path = os.path.join(root, dir_name)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        features = extract_features(img_path)
                        
                        logo = Logo(
                            name=dir_name,
                            path=img_path,
                            feature_vector=features
                        )
                        db.session.add(logo)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    db.session.commit()
    print("Dataset processing completed!")
