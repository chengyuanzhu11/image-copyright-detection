# 图像版权检测系统

基于 `Flask + TensorFlow/ResNet50` 的毕业设计项目，面向 Logo 版权检测与 AI 生成图像初步识别场景。项目包含后端 API、前端页面、Logo 数据导入脚本、模型训练脚本，以及论文文档。

## 仓库说明

当前 GitHub 仓库托管的是“轻量代码版”项目：

- 包含源码、前端页面、轻量元数据和论文文档。
- 不包含大体积数据集、训练产物、运行缓存、日志和本地虚拟环境。

大文件和数据恢复方式见 [DATASET.md](DATASET.md)。

## 主要功能

- 上传图片并进行 Logo 相似度检测。
- 返回相似品牌、类别和详细检测结果。
- 对图片进行 AI 生成概率初步分析。
- 导入 `Logo-2K+` 数据集并生成特征库。
- 训练基于 `ResNet50` 的 Logo 分类模型。
- 管理 Logo 元数据、类别和特征重建。

## 技术栈

- 后端：Python、Flask、Flask-CORS、Flask-RESTX、SQLAlchemy
- 深度学习与图像处理：TensorFlow 2.8、ResNet50、NumPy、Pillow、OpenCV、scikit-learn
- 前端：HTML、CSS、JavaScript
- 数据存储：JSON、Pickle、SQLite

## 目录结构

```text
backend/                         Flask API、模型脚本、数据导入与训练脚本
frontend/                        前端页面与静态资源
data/                            本地数据集目录（已忽略，不上传）
models/                          本地模型产物目录（已忽略，不上传）
logo_data.json                   Logo 元数据
paper_text.txt                   论文文本提取结果
基于机器学习的图像版权检测系统.pdf  论文文档
```

## 快速开始

建议始终在项目根目录执行命令，避免模型、日志和特征文件写入不同位置。

1. 创建并激活虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. 安装依赖

```powershell
pip install -r backend/requirements.txt
pip install -r requirements.txt
pip install matplotlib tqdm
```

说明：

- 当前仓库依赖文件仍是分散状态，所以上面用了组合安装。
- 首次运行 TensorFlow/ResNet50 时，可能会下载预训练权重。

3. 启动完整应用

```powershell
python backend/simple_app.py
```

默认访问地址：

```text
http://localhost:5000/
```

如果你只想运行较精简的接口版本，可以尝试：

```powershell
python backend/app.py
```

## 主要接口

`backend/simple_app.py` 中包含的主要路由如下：

- `POST /detect`：Logo 相似度检测
- `GET /detailed_result`：查看详细检测结果
- `POST /detect_ai_generated`：AI 生成图片检测
- `POST /add_logo`：新增 Logo
- `GET /logos`：获取 Logo 列表
- `GET /categories`：获取类别列表
- `POST /train_model`：启动模型训练
- `GET /model_status`：查询模型训练状态
- `POST /import_dataset`：导入数据集
- `POST /reset_features`：重建特征库

## 数据与模型说明

仓库中没有包含以下本地大文件内容：

- `data/`
- `models/`
- `backend/models/`
- `logo_features.pkl`
- `*.pkl.bak`
- `*.db`
- `*.log`
- `temp_uploads/`

另外，当前仓库中的 `logo_data.json` 是已有元数据文件，但其中的 `image_path` 字段可能来自旧机器的本地路径。迁移到新环境后，建议按 [DATASET.md](DATASET.md) 的步骤重新生成一份，保证路径和你当前机器一致。

## 相关文档

- [DATASET.md](DATASET.md)
- [backend/README.md](backend/README.md)
- [基于机器学习的图像版权检测系统.pdf](基于机器学习的图像版权检测系统.pdf)
