from flask import Flask, request, jsonify
import os
import numpy as np
from ai_detector import detect_ai_generated
import logging
import time
import uuid
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 确保临时文件夹存在
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
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

@app.route('/detect_ai_generated', methods=['POST'])
def detect_ai_generated_endpoint():
    """检测图片是否为AI生成的测试端点"""
    try:
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
            
            # 执行检测
            logger.info(f"开始AI检测: {temp_path}")
            score, analysis = detect_ai_generated(temp_path)
            logger.info(f"检测完成，得分: {score}")
                
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
            
            # 使用convert_numpy_types函数处理结果，确保没有NumPy类型
            result = convert_numpy_types(result)
            
            return jsonify(result)
                
        except Exception as e:
            logger.error(f"AI检测处理失败: {e}")
            import traceback
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
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI检测测试</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #4361ee; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            button { background: #4361ee; color: white; border: none; padding: 10px 15px; cursor: pointer; }
            #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; min-height: 100px; }
        </style>
    </head>
    <body>
        <h1>AI生成图片检测测试服务</h1>
        <div class="form-group">
            <label for="imageFile">选择图片：</label>
            <input type="file" id="imageFile" accept="image/*">
        </div>
        <div class="form-group">
            <label for="threshold">检测阈值 (%)：</label>
            <input type="range" id="threshold" min="10" max="90" value="50">
            <span id="thresholdValue">50</span>%
        </div>
        <button id="detectBtn">开始检测</button>
        
        <div id="result">
            <p>检测结果将显示在这里</p>
        </div>
        
        <script>
            document.getElementById('threshold').addEventListener('input', function() {
                document.getElementById('thresholdValue').textContent = this.value;
            });
            
            document.getElementById('detectBtn').addEventListener('click', function() {
                const fileInput = document.getElementById('imageFile');
                const threshold = document.getElementById('threshold').value;
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files || fileInput.files.length === 0) {
                    resultDiv.innerHTML = '<p style="color: red;">请选择图片文件</p>';
                    return;
                }
                
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('image', file);
                formData.append('threshold', threshold);
                
                resultDiv.innerHTML = '<p>正在检测中，请稍候...</p>';
                
                fetch('/detect_ai_generated', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP错误: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('收到服务器响应:', data);
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<p style="color: red;">错误: ${data.error}</p>`;
                        return;
                    }
                    
                    const probability = data.ai_generated_probability * 100;
                    const conclusion = data.conclusion;
                    
                    let html = `
                        <h3>检测结论: ${conclusion}</h3>
                        <p>AI生成概率: ${probability.toFixed(2)}%</p>
                        <p>检测阈值: ${data.threshold}%</p>
                    `;
                    
                    if (data.suspicious_features && data.suspicious_features.length > 0) {
                        html += '<h4>发现可疑特征:</h4><ul>';
                        data.suspicious_features.forEach(feature => {
                            html += `<li>${feature}</li>`;
                        });
                        html += '</ul>';
                    }
                    
                    resultDiv.innerHTML = html;
                })
                .catch(error => {
                    console.error('请求失败:', error);
                    resultDiv.innerHTML = `<p style="color: red;">检测失败: ${error.message}</p>`;
                });
            });
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("启动测试服务器，访问 http://localhost:5001/ 测试AI检测功能")
    app.run(host='0.0.0.0', port=5001, debug=True) 