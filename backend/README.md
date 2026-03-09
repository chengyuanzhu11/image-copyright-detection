# Logo版权检测系统

基于深度学习的Logo版权检测系统，利用ResNet50预训练模型实现高准确度的Logo识别和匹配。

## 功能特点

1. **深度学习Logo识别**：
   - 使用ResNet50为基础构建自定义模型
   - 支持模型微调和迁移学习
   - 高精度的Logo分类

2. **特征匹配和相似度计算**：
   - 提取Logo的深度特征
   - 基于余弦相似度的匹配算法
   - 兼容新增和未见过的Logo

3. **完整的API接口**：
   - Logo检测和识别
   - 模型训练和状态监控
   - 特征库管理和数据导入

## 系统架构

- **前端**：Web界面，用于上传图像和显示检测结果
- **后端**：Flask API服务，处理请求和运行模型
- **模型**：基于ResNet50的深度学习模型

## 安装和配置

### 环境要求

- Python 3.6+
- TensorFlow 2.0+
- Flask
- 其他依赖项，见`requirements.txt`

### 安装步骤

1. 克隆代码库：
   ```
   git clone <仓库地址>
   cd <项目目录>/backend
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 数据目录结构：
   ```
   data/
     train_and_test/
       train/
         类别1/
           图像文件...
         类别2/
           图像文件...
       test/
         测试图像...
   ```

## 使用方法

### 启动服务器

```
python simple_app.py
```

服务器默认在 http://localhost:8080 运行。

### 训练模型

#### 使用API

```
python test_api.py --train default
```

或者指定自定义数据集路径：

```
python test_api.py --train 数据集路径
```

#### 使用训练脚本

```
python train_model.py --data_dir data/train_and_test/train --epochs 30 --batch_size 32
```

### 测试Logo检测

```
python test_api.py --detect 图像路径
```

### 查看模型状态

```
python test_api.py --status
```

## API接口说明

### 1. Logo检测

**端点**: `POST /detect`

**参数**:
- `image`: 图像文件（multipart/form-data）

**响应**:
```json
{
  "result": "检测到Logo: Brand名称, 置信度: 0.95",
  "detection_id": "唯一ID",
  "method": "trained_model"
}
```

### 2. 获取详细结果

**端点**: `GET /detailed_result?id=检测ID`

**响应**:
```json
{
  "timestamp": 1650000000,
  "filename": "image.jpg",
  "main_result": "检测到Logo: Brand名称, 置信度: 0.95",
  "model_prediction": {
    "class": "Brand名称",
    "confidence": 0.95
  },
  "similar_logos": [...],
  "detection_method": "trained_model"
}
```

### 3. 训练模型

**端点**: `POST /train_model`

**参数**:
```json
{
  "dataset_path": "data/train_and_test/train",
  "epochs": 30,
  "batch_size": 32
}
```

**响应**:
```json
{
  "status": "started",
  "message": "模型训练已启动"
}
```

### 4. 获取模型状态

**端点**: `GET /model_status`

**响应**:
```json
{
  "training_status": {
    "is_training": false,
    "progress": 100,
    "message": "模型训练完成"
  },
  "has_trained_model": true,
  "model_info": {
    "class_count": 10,
    "feature_count": 500
  }
}
```

## 训练和数据集说明

### 数据集格式

训练数据集需要按照以下格式组织：
- 每个类别一个文件夹
- 文件夹名称为类别名称
- 每个文件夹中包含该类别的Logo图像

例如：
```
train/
  Nike/
    nike1.jpg
    nike2.jpg
  Adidas/
    adidas1.jpg
    adidas2.jpg
```

### 训练过程

1. 数据准备：清洗和组织数据集
2. 模型初始训练：冻结ResNet50基础层
3. 模型微调：解冻部分层进行微调
4. 特征提取：为所有Logo提取特征向量
5. 模型保存：保存训练好的模型和特征数据

## 注意事项

1. 初次加载时会下载ResNet50预训练权重
2. 模型训练可能需要较长时间，建议在GPU环境下运行
3. 可以通过API监控训练进度
4. 测试图像需要是PNG、JPG或JPEG格式

## 故障排除

1. **模型加载失败**
   - 检查模型文件路径
   - 确保TensorFlow版本兼容

2. **训练失败**
   - 检查数据集格式和路径
   - 确保每个类别有足够的样本

3. **检测结果不准确**
   - 增加训练数据多样性
   - 调整训练超参数
   - 检查图像预处理步骤 