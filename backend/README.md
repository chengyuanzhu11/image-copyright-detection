# Backend 说明

`backend/` 目录包含 Flask 接口、Logo 检测逻辑、AI 图片检测逻辑、数据导入脚本和模型训练脚本。

## 推荐启动方式

请在项目根目录执行：

```powershell
python backend/simple_app.py
```

默认服务地址：

```text
http://localhost:5000/
```

`simple_app.py` 是当前功能更完整的入口，包含：

- Logo 检测
- AI 图片检测
- 详细结果查询
- Logo 管理
- 数据集导入
- 模型训练状态查询

如果只需要较轻量的接口版本，可以尝试：

```powershell
python backend/app.py
```

## 常用脚本

### 1. 导入 Logo-2K+ 数据集

```powershell
python backend/import_logos.py data/Logo-2K+
```

作用：

- 遍历数据集目录
- 提取每张 Logo 图片的特征
- 生成 `logo_features.pkl`
- 更新 `logo_data.json`

### 2. 训练 Logo 模型

```powershell
python backend/train_model.py --data_dir data/train_and_test/train --epochs 30 --batch_size 32
```

作用：

- 准备训练数据
- 训练基于 ResNet50 的分类模型
- 输出模型文件和训练历史

## 主要接口

- `POST /detect`
- `GET /detailed_result`
- `POST /detect_ai_generated`
- `POST /add_logo`
- `GET /logos`
- `GET /categories`
- `POST /train_model`
- `GET /model_status`
- `POST /import_dataset`
- `POST /reset_features`
- `DELETE /delete_logo/<logo_id>`

## 路径与生成文件说明

项目里部分脚本使用相对路径，所以执行目录会影响生成文件位置。为了避免混乱，建议统一从仓库根目录执行命令。

常见本地产物包括：

- `models/`
- `logo_features.pkl`
- `logo_detection.db`
- `app.log`
- `training.log`

这些文件默认都不建议提交到 GitHub。
