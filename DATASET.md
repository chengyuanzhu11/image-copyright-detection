# 数据与模型说明

这个仓库没有把完整数据集和训练产物推到 GitHub。原因很简单：数据量大、文件数多，而且模型与特征文件属于典型的本地产物，不适合进 Git 历史。

## 未纳入仓库的内容

以下内容默认只保留在本地：

- `data/`
- `models/`
- `backend/models/`
- `logo_features.pkl`
- `*.pkl.bak`
- `*.db` / `*.sqlite`
- `app.log`、`training.log` 等日志文件
- `temp_uploads/`、`backend/temp/` 等运行时目录

## 建议的执行约定

建议始终在项目根目录执行命令，例如：

```powershell
python backend/simple_app.py
python backend/import_logos.py data/Logo-2K+
python backend/train_model.py --data_dir data/train_and_test/train
```

这样做的原因是：

- 生成文件会更集中，便于管理。
- 不会出现一部分文件写到根目录、另一部分写到 `backend/` 的情况。
- 相对路径更容易控制。

## 场景 1：导入 Logo-2K+ 数据集

`backend/import_logos.py` 的目标是遍历数据集，提取特征，并生成：

- `logo_features.pkl`
- `logo_data.json`

推荐的数据目录结构：

```text
data/
  Logo-2K+/
    Accessories/
      BrandA/
        1.jpg
        2.jpg
    Food/
      BrandB/
        1.png
```

从项目根目录执行：

```powershell
python backend/import_logos.py data/Logo-2K+
```

也可以使用包装脚本：

```powershell
python backend/run_import.py
```

但这个包装脚本依赖脚本默认路径，更推荐直接传显式路径给 `import_logos.py`。

## 场景 2：训练 Logo 分类模型

`backend/train_model.py` 用于训练基于 `ResNet50` 的 Logo 分类模型。

推荐训练目录结构：

```text
data/
  train_and_test/
    train/
      BrandA/
        a.jpg
        b.jpg
      BrandB/
        a.jpg
```

从项目根目录执行：

```powershell
python backend/train_model.py --data_dir data/train_and_test/train --epochs 30 --batch_size 32
```

训练产物通常会写入本地 `models/` 目录，例如：

- `models/logo_detection_model.h5`
- `models/class_mapping.json`
- `models/training_history.json`
- `models/feature_vectors.pkl`

## `logo_data.json` 的可移植性说明

当前仓库里已经提交了一份 `logo_data.json`，但它的 `image_path` 字段包含旧机器导出时的本地路径。换机器后，这些路径很可能失效。

如果你要在新环境中完整恢复项目，建议这样做：

1. 把数据集恢复到 `data/Logo-2K+` 或你自己的本地目录。
2. 从项目根目录执行：

```powershell
python backend/import_logos.py data/Logo-2K+
```

3. 用新生成的 `logo_data.json` 和 `logo_features.pkl` 覆盖旧的本地文件。
