from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import time
import uuid
import traceback
from werkzeug.utils import secure_filename
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)

# 允许所有跨域请求
CORS(app, resources={r"/*": {"origins": "*"}})

# 确保临时文件夹存在
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/detect_ai_generated', methods=['POST', 'OPTIONS'])
def detect_ai_generated():
    """处理AI图片检测请求"""
    # 处理OPTIONS请求（预检请求）
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        logger.info("收到AI检测请求")
        
        # 检查是否有图片文件
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
            
            # 获取阈值参数
            threshold = request.form.get('threshold', '50')
            try:
                threshold_value = float(threshold) / 100
            except (ValueError, TypeError):
                threshold_value = 0.5
            
            # 导入并调用AI检测模块
            try:
                from ai_detector import detect_ai_generated
                logger.info(f"开始AI检测: {temp_path}")
                score, analysis = detect_ai_generated(temp_path)
                logger.info(f"检测完成，得分: {score}")
            except Exception as e:
                logger.error(f"AI检测失败: {e}")
                return jsonify({
                    'error': '无法执行AI检测',
                    'details': str(e)
                }), 500
            
            # 处理结果
            result = {
                'detection_id': str(uuid.uuid4()),
                'ai_generated_probability': float(score),
                'conclusion': "AI生成" if score > threshold_value else "真实照片",
                'analysis': analysis,
                'suspicious_features': analysis.get('suspicious_features', []),
                'timestamp': time.time(),
                'threshold': threshold_value * 100
            }
            
            # 转换NumPy类型
            result = convert_numpy_types(result)
            
            return jsonify(result)
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """简单的首页"""
    return """
    <html>
        <head><title>AI检测API</title></head>
        <body>
            <h1>AI检测API服务器</h1>
            <p>使用POST请求访问 /detect_ai_generated 接口</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        # 检查AI检测模块
        from ai_detector import detect_ai_generated
        print("✅ AI检测器模块加载成功")
    except Exception as e:
        print(f"❌ 警告: AI检测器模块加载失败: {e}")
    
    # 启动服务器
    print("✅ 启动简化版API服务器在端口5001")
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True) 