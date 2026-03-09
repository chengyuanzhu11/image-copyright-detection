document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    const fileSelect = document.getElementById('fileSelect');
    const preview = document.getElementById('preview');
    const uploadStatus = document.getElementById('uploadStatus');
    const result = document.getElementById('result');
    const alertMessage = document.getElementById('alertMessage');
    
    // 确保元素存在再添加事件处理
    if (fileSelect && fileElem) {
        // 处理文件选择器点击
        fileSelect.addEventListener('click', function(e) {
            e.preventDefault();
            fileElem.click();
        });
        
        // 处理文件选择
        fileElem.addEventListener('change', function() {
            handleFiles(this.files);
        });
    }
    
    // 处理拖放区域事件
    if (dropArea) {
        // 阻止默认行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // 高亮拖放区域
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('drop-area-highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('drop-area-highlight');
        }
        
        // 处理拖放的文件
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
    }
    
    // 处理文件上传
    function handleFiles(files) {
        if (!files || files.length === 0) return;
        
        const file = files[0];
        
        // 检查是否为图片
        if (!file.type.match('image.*')) {
            showAlert('请选择图片文件', 'warning');
            return;
        }
        
        // 显示预览
        if (preview) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        
        // 上传图片
        uploadImage(file);
    }
    
    // 上传图片到服务器
    function uploadImage(file) {
        // 清空结果区域
        if (result) result.innerHTML = '';
        if (alertMessage) alertMessage.style.display = 'none';
        
        // 显示加载状态
        if (uploadStatus) uploadStatus.style.display = 'block';
        
        // 创建FormData
        const formData = new FormData();
        formData.append('image', file);
        
        // 创建图片的URL，用于传递给详细结果页面
        const imageUrl = URL.createObjectURL(file);
        
        // 发送请求到后端
        fetch('http://localhost:5000/detect', {
            method: 'POST',
            body: formData,
            mode: 'cors',
            credentials: 'omit'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络响应错误');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                // 获取检测ID，如果没有，则生成一个临时ID
                const detectionId = data.detection_id || 'temp_' + new Date().getTime();
                
                // 获取结果文本
                const resultText = data.result || (typeof data === 'string' ? data : JSON.stringify(data));
                
                // 构建结果HTML，始终包含查看详细结果的链接
                const resultHtml = `
                    <div class="alert alert-info">
                        <h4>检测结果</h4>
                        <p>${resultText}</p>
                        <a href="detailed_result.html?id=${detectionId}&image=${encodeURIComponent(imageUrl)}" class="btn btn-outline-primary btn-sm mt-2" target="_blank">
                            查看详细分析 <i class="fas fa-chart-bar"></i>
                        </a>
                    </div>
                `;
                
                if (result) result.innerHTML = resultHtml;
                
                // 如果后端没有返回detection_id，可以将结果临时存储在localStorage中
                if (!data.detection_id) {
                    try {
                        const tempResult = {
                            main_result: resultText,
                            matches: [{
                                name: "未能获取完整匹配数据",
                                similarity: 0.0,
                                category: "未知"
                            }]
                        };
                        localStorage.setItem('temp_detection_' + detectionId, JSON.stringify(tempResult));
                    } catch (e) {
                        console.error('无法存储临时结果', e);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('检测失败，请稍后重试: ' + error.message, 'danger');
        })
        .finally(() => {
            if (uploadStatus) uploadStatus.style.display = 'none';
        });
    }
    
    // 显示提示信息
    function showAlert(message, type) {
        if (!result) return;
        
        let detailsLink = '';
        
        if (fileElem && fileElem.files && fileElem.files[0]) {
            const file = fileElem.files[0];
            const imageUrl = URL.createObjectURL(file);
            const tempId = 'temp_' + new Date().getTime();
            
            // 存储临时错误信息
            try {
                const tempResult = {
                    main_result: message,
                    matches: [{
                        name: "处理失败",
                        similarity: 0.0,
                        category: "错误"
                    }]
                };
                localStorage.setItem('temp_detection_' + tempId, JSON.stringify(tempResult));
            } catch (e) {
                console.error('无法存储临时结果', e);
            }
            
            detailsLink = `
                <br><br>
                <a href="detailed_result.html?id=${tempId}&image=${encodeURIComponent(imageUrl)}" class="btn btn-outline-info btn-sm" target="_blank">
                    查看详细信息 <i class="fas fa-search"></i>
                </a>
            `;
        }
        
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : type === 'warning' ? 'exclamation-circle' : 'info-circle'}"></i> 
                ${message}${detailsLink}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        result.innerHTML = alertHtml;
        if (alertMessage) alertMessage.style.display = 'none';
    }
    
    // 添加拖放区域的高亮样式
    if (dropArea) {
        // 添加高亮样式
        document.head.insertAdjacentHTML('beforeend', `
            <style>
                .drop-area-highlight {
                    background-color: rgba(67, 97, 238, 0.15) !important;
                    border: 2px dashed #3a7bd5 !important;
                    transform: scale(1.02);
                }
                .drop-area-ready {
                    position: relative;
                    transition: all 0.3s ease;
                }
            </style>
        `);
        
        // 初始化样式
        dropArea.classList.add('drop-area-ready');
    }
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const similarityPercent = (result.similarity * 100).toFixed(1);
    
    resultDiv.innerHTML = `
        <div class="alert ${result.similarity > 0.7 ? 'alert-success' : 'alert-warning'}">
            <h4>检测结果</h4>
            <p>最相似Logo：${result.match}</p>
            <p>相似度：${similarityPercent}%</p>
            <p>判定阈值：${result.threshold * 100}%</p>
        </div>
    `;
} 