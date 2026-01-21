# PyTorch AlexNet

[切换到英文](README.md) | [中文](#中文)

---

# PyTorch AlexNet

用PyTorch实现的经典AlexNet，在Imagenette数据集上进行图像分类训练。

## 简介

这是一个AlexNet架构的PyTorch实现，在Imagenette数据集（ImageNet的10个类别子集）上进行训练。

## 项目特点

- **框架**：PyTorch
- **数据集**：Imagenette（ImageNet的10类子集）
- **监控**：TensorBoard可视化损失和准确率
- **指标**：支持Top-1和Top-5准确率跟踪
- **模型**：经典AlexNet架构，使用ReLU激活函数和Dropout正则化

## 项目结构

```
.
├── main.py              # 训练脚本
├── model.py             # AlexNet模型定义
├── utils.py             # 工具函数和指标
├── datasets/
│   └── imagenette2/     # Imagenette数据集（train/val文件夹）
├── checkpoints/         # 保存的模型权重
└── runs/                # TensorBoard日志
    └── alexnet/
```

## 环境要求

```
python>=3.7
torch
torchvision
tqdm
```

推荐使用CUDA支持，但也支持CPU训练。

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd alexnet_pytorch
```

2. 安装依赖：
```bash
pip install torch torchvision tqdm
```

## 数据集准备

1. 从 [fastai/imagenette](https://github.com/fastai/imagenette) 下载Imagenette数据集

2. 解压到datasets文件夹：
```bash
tar -xzf imagenette2.tgz -C datasets/
```

目录结构应该如下所示：
```
datasets/
└── imagenette2/
    ├── train/
    │   ├── n01440764/
    │   ├── n02102040/
    │   └── ... （还有8个类别）
    └── val/
        ├── n01440764/
        ├── n02102040/
        └── ... （还有8个类别）
```

## 训练

### 基本用法

运行训练脚本：
```bash
python main.py
```

### 命令行参数

- `--dataset_root_dir`：数据集目录路径（默认：`datasets/imagenette2`）
- `--epochs`：训练轮数（默认：`50`）
- `--batch-size`：批次大小（默认：`64`）
- `--lr`：学习率（默认：`0.001`）
- `--num_classes`：类别数（默认：`1000`用于完整ImageNet，Imagenette使用`10`）

### 示例

```bash
python main.py --epochs 100 --batch-size 128 --lr 0.001 --num_classes 10
```

## 使用TensorBoard监控训练

训练脚本会自动将指标记录到TensorBoard中。要进行可视化：

1. 启动TensorBoard：
```bash
tensorboard --logdir=runs
```

2. 在浏览器中打开：`http://localhost:6006`

可以跟踪以下内容：
- 训练损失和准确率
- 验证损失和准确率
- 最佳模型性能

## 模型架构

AlexNet模型包括：

**特征提取**：
- 5个卷积层，使用ReLU激活函数
- 3个最大池化层
- 输入：224×224 RGB图像
- 输出：256个特征图（6×6）

**分类**：
- 3个全连接层
- 2个Dropout层用于正则化
- 输出：num_classes个预测

## 主要功能

✅ 基于验证准确率自动保存最佳模型  
✅ 集成TensorBoard实时监控  
✅ 支持CPU和GPU训练  
✅ Top-1和Top-5准确率指标  
✅ 交叉熵损失跟踪  
✅ 清晰的模块化代码结构  

## 输出

训练完成后：
- **模型检查点**保存在`checkpoints/`目录中
- **最佳模型**在验证准确率提高时自动保存
- **TensorBoard日志**保存在`runs/alexnet/`中用于可视化

## 注意事项

- 对于Imagenette数据集，通常使用`--num_classes 10`
- 默认学习率（0.001）与Adam优化器配合效果很好
- 训练进度和指标同时记录到控制台和TensorBoard
- 每当验证准确率提高时，最佳模型就会自动保存

## 许可证

该项目仅供教育目的使用。

## 参考文献

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS.
- [Imagenette数据集](https://github.com/fastai/imagenette)
