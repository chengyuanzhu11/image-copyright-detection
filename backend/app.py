from flask import Flask, Blueprint, request, jsonify, send_from_directory
from model import detect_logo_similarity
import logging
from config import Config
from flask_restx import Api, Resource, fields
import os
from pathlib import Path
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import time
import uuid
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='', static_folder='../frontend')
app.config.from_object(Config)

# 配置CORS
CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     expose_headers=["Content-Type", "Content-Length"]
)

# 在每个响应前添加CORS头
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length')
    # 确保OPTIONS请求能顺利通过
    if request.method == 'OPTIONS':
        return response
    return response

api_bp = Blueprint('api', __name__)
api = Api(api_bp, version='1.0', title='图像版权检测API',
          description='基于深度学习的图像版权检测系统API',
          doc='/api-docs')
app.register_blueprint(api_bp, url_prefix='/api')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 确保临时文件夹存在
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# 更新配置
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

detect_model = api.model('DetectRequest', {
    'image': fields.Raw(description='上传的图片文件', required=True)
})

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logo_detection.db'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Logo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    feature_vector = db.Column(db.PickleType, nullable=False)
    
    def __repr__(self):
        return f'<Logo {self.name}>'

# 确保数据库目录存在
db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
if db_path and db_path != '':
    db_dir = os.path.dirname(db_path)
    if db_dir and db_dir != '':
        os.makedirs(db_dir, exist_ok=True)
        print(f"✅ 确保数据库目录存在: {db_dir}")

with app.app_context():
    db.create_all()

LOGO_DB_PATH = Path("logo_features.db")
if LOGO_DB_PATH.exists():
    LOGO_FEATURES = joblib.load(LOGO_DB_PATH)
else:
    LOGO_FEATURES = {}

# 存储检测结果
DETECTION_RESULTS = {}

@app.route('/detect', methods=['POST'])
def detect():
    """处理图片版权检测请求"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传图片文件'}), 400
                
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        allowed_extensions = Config.ALLOWED_EXTENSIONS
        if '.' not in image.filename or image.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type'}), 400

        temp_path = os.path.join(app.config['TEMP_FOLDER'], image.filename)
        image.save(temp_path)
        
        result = detect_logo_similarity(temp_path)
        return jsonify({'result': result})
        
    except Exception as e:
        app.logger.error(f"Error during detection: {e}")
        return jsonify({'error': 'Detection failed, please try again later'}), 500

@app.route('/detect_ai_generated', methods=['POST'])
def detect_ai_generated():
    """检测图片是否为AI生成"""
    try:
        # 导入AI检测模块
        try:
            from ai_detector import detect_ai_generated as ai_detect
        except ImportError as e:
            app.logger.error(f"导入AI检测模块失败: {e}")
            ai_detect = None
        
        if 'image' not in request.files:
            app.logger.warning("未上传图片文件")
            return jsonify({'error': '未上传图片文件'}), 400
                
        image = request.files['image']
        if image.filename == '':
            app.logger.warning("空文件名")
            return jsonify({'error': '未选择任何文件'}), 400
        
        if not allowed_file(image.filename):
            allowed_extensions = ', '.join(ALLOWED_EXTENSIONS)
            app.logger.warning(f"不支持的文件类型: {image.filename}")
            return jsonify({'error': f'不支持的文件类型。请上传 {allowed_extensions} 格式的图片'}), 400

        # 生成唯一的临时文件名
        temp_filename = f"ai_check_{int(time.time())}_{secure_filename(image.filename)}"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        try:
            # 保存上传的图片
            image.save(temp_path)
            app.logger.info(f"图片已保存到: {temp_path}")
            
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
                app.logger.info(f"使用阈值: {threshold}%")
            except (ValueError, TypeError):
                app.logger.warning(f"无效的阈值参数: {threshold}，使用默认阈值")
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
                    # 执行检测
                    app.logger.info(f"开始AI检测: {temp_path}")
                    score, analysis = ai_detect(temp_path)
                    app.logger.info(f"检测完成，得分: {score}")
                    
                except Exception as e:
                    # 如果导入失败，使用基本图像分析
                    app.logger.warning(f"AI检测模块导入失败，使用基本分析: {e}")
                    score, analysis = fallback_detection(temp_path)
                    
                # 处理结果 - 确保转换为普通Python类型避免序列化问题
                ai_score = float(score)
                
                # 判断是否是证件照，已在AI检测模块中处理
                is_id_photo = analysis.get('is_id_photo', False)
                if is_id_photo:
                    app.logger.info(f"检测到可能的证件照，分数: {ai_score}")
                
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
                
                # 使用convert_numpy_types函数处理结果，确保没有NumPy类型
                result = convert_numpy_types(result)
                
                return jsonify(result)
                
            except Exception as e:
                app.logger.error(f"AI检测处理失败: {e}")
                app.logger.error(traceback.format_exc())
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
                    app.logger.info(f"临时文件已删除: {temp_path}")
            except Exception as e:
                app.logger.warning(f"删除临时文件失败: {e}")
            
    except Exception as e:
        app.logger.error(f"AI检测请求处理失败: {e}")
        return jsonify({'error': str(e)}), 500

def fallback_detection(image_path):
    """当AI模型不可用时的降级检测方法"""
    app.logger.info(f"使用降级检测模式: {image_path}")
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
        app.logger.error(f"降级检测也失败: {e}")
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

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({'error': 'Resource not found'}), 404

if __name__ == '__main__':
    # 在启动前检查AI检测器是否可用
    try:
        from ai_detector import detect_ai_generated
        print("✅ AI检测器模块导入成功")
        # 确认临时文件夹存在
        print(f"✅ 临时文件夹设置为: {TEMP_FOLDER}")
        if not os.path.exists(TEMP_FOLDER):
            os.makedirs(TEMP_FOLDER)
            print(f"✅ 创建临时文件夹: {TEMP_FOLDER}")
        else:
            print(f"✅ 临时文件夹已存在: {TEMP_FOLDER}")
    except Exception as e:
        print(f"❌ 警告: AI检测器初始化失败: {e}")

    print("✅ 准备启动应用，监听端口5000...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
