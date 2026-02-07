# Attention Mechanisms for Image Segmentation

深度学习图像分割项目，包含三种注意力机制的实现和训练脚本。

## 项目结构

```
Jill-z-arch/
├── Coordinate_Attention/     # 坐标注意力机制
├── Rotate_to_Attend/          # 旋转注意力（Triplet Attention）
├── Sparge_Attention/          # 稀疏注意力机制
└── README.md
```

---

## 环境配置

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ / CentOS 7+)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (推荐)
- **内存**: 16GB+ RAM
- **存储**: 50GB+ 可用空间

### 1. 安装 Conda/Miniforge

```bash
# 下载并安装 Miniforge3
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 初始化 conda
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash
```

### 2. 创建 Python 环境

```bash
# 创建 Python 3.10 环境
conda create -n py310_local python=3.10 -y
conda activate py310_local
```

### 3. 安装 PyTorch 和依赖

```bash
# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或者 CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CPU only (不推荐，训练会很慢)
conda install pytorch torchvision cpuonly -c pytorch -y

# 安装其他依赖
conda install pillow numpy -y
pip install tqdm
```

### 4. 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

预期输出：
```
PyTorch: 2.x.x
CUDA available: True
torchvision: 0.x.x
```

### 5. 安装 tmux（可选，用于后台训练）

```bash
# Ubuntu/Debian
sudo apt-get install tmux -y

# CentOS/RHEL
sudo yum install tmux -y
```

---

## 数据集准备

### 数据集结构

项目需要以下数据结构：

```
dataset/
├── DATASET/
│   └── Segmentation/
│       ├── Images/
│       │   ├── img001.jpg
│       │   ├── img002.jpg
│       │   └── ...
│       └── Labels/
│           ├── img001.png
│           ├── img002.png
│           └── ...
├── ds2_train_labeled.txt
└── ds2_val_labeled.txt
```

### 标注文件格式

`ds2_train_labeled.txt` 和 `ds2_val_labeled.txt` 格式：

```
Images/img001.jpg Labels/img001.png
Images/img002.jpg Labels/img002.png
...
```

每行包含两列，用空格分隔：
- 第一列：相对于 `IMAGE_ROOT` 的图像路径
- 第二列：相对于 `IMAGE_ROOT` 的标签路径

### 修改数据路径

在运行脚本前，需要修改脚本中的数据路径。打开 `run_*.sh` 文件，修改以下变量：

```bash
IMAGE_ROOT="/path/to/your/dataset/DATASET/Segmentation"
TRAIN_TXT="/path/to/your/ds2_train_labeled.txt"
VAL_TXT="/path/to/your/ds2_val_labeled.txt"
```

---

## 使用说明

### Coordinate_Attention - 坐标注意力

结合坐标注意力和 Agent Attention 的图像分割模型。

#### 修改配置

编辑 `Coordinate_Attention/run_Coordinate_Attention.sh`，修改：

```bash
# 数据路径
IMAGE_ROOT="/your/path/to/Segmentation"
TRAIN_TXT="/your/path/to/ds2_train_labeled.txt"
VAL_TXT="/your/path/to/ds2_val_labeled.txt"

# Conda 环境路径
CONDA_ENV="/home/your_username/miniforge3/envs/py310_local"

# 训练参数（可选）
NUM_CLASSES=4          # 类别数量
EPOCHS=100            # 训练轮数
BATCH_SIZE=16         # 批次大小
LR=0.025              # 学习率
```

#### 运行训练

```bash
cd Coordinate_Attention
./run_Coordinate_Attention.sh
```

#### 查看训练日志

```bash
# 实时查看日志
tmux attach -t train_coord_agent_attn

# 或者查看日志文件
tail -f logs_tmux/train_*.log

# 退出 tmux 会话（不终止训练）
# 按 Ctrl+B，然后按 D
```

---

### Rotate_to_Attend - 旋转注意力

基于 Triplet Attention（三维旋转注意力）的分割模型。

#### 修改配置

编辑 `Rotate_to_Attend/run_Rotate_to_Attend_tmux.sh`：

```bash
# 数据路径
IMAGE_ROOT="/your/path/to/Segmentation"
TRAIN_TXT="/your/path/to/ds2_train_labeled.txt"
VAL_TXT="/your/path/to/ds2_val_labeled.txt"

# Conda 环境路径
CONDA_ENV="/home/your_username/miniforge3/envs/py310_local"

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LR=0.025

# Triplet Attention 参数
USE_TRIPLET_ATTN=1
TRIPLET_KERNEL=7
ATTN_ORDER="triplet_then_agent"  # 注意力顺序
```

#### 运行训练

```bash
cd Rotate_to_Attend
./run_Rotate_to_Attend_tmux.sh
```

#### 查看训练

```bash
# 附加到 tmux 会话
tmux attach -t train_rotate_to_attend

# 查看日志
tail -f logs_tmux/train_*.log
```

---

### Sparge_Attention - 稀疏注意力

稀疏注意力机制，优化计算效率。

#### 修改配置

编辑 `Sparge_Attention/run_Sparge_Agent_Attention_tmux.sh`：

```bash
# 数据路径（在脚本内部 RUN_EOF heredoc 中）
# 找到以下行并修改：
  --image_root "/your/path/to/Segmentation" \
  --train_txt  "/your/path/to/ds2_train_labeled.txt" \
  --val_txt    "/your/path/to/ds2_val_labeled.txt" \

# Conda 环境路径
CONDA="/home/your_username/miniforge3/envs/py310_local"

# 稀疏注意力参数
--use_sparge_attn \
--sparge_topk 64 \
--attn_order "sparge_then_agent"
```

#### 运行训练

```bash
cd Sparge_Attention
./run_Sparge_Agent_Attention_tmux.sh
```

#### 查看训练

```bash
# 附加到 tmux 会话
tmux attach -t train_sparge_agent_attn

# 查看日志
tail -f logs_tmux/train_*.log
```

---

## 输出文件

训练完成后，每个项目会在 `output/` 目录生成：

```
output/
├── best.pt              # 验证集上最佳模型
├── last.pt              # 最后一轮的模型（部分项目）
├── run_config.json      # 训练配置
└── bad_images.txt       # 损坏的图像列表（如有）
```

### 使用训练好的模型

```python
import torch

# 加载模型
model = torch.load('output/best.pt')
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

---

## 常见问题

### 1. CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小 batch_size
BATCH_SIZE=8  # 或更小

# 减小图像分辨率
# 或者使用梯度累积
```

### 2. ImportError: urllib cannot be found

**症状**: 启动时提示 urllib 错误

**解决方案**:
```bash
# 重新安装 PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall -y
```

### 3. tmux 会话已存在

**症状**: `tmux session already exists`

**解决方案**:
```bash
# 杀死旧会话
tmux kill-session -t train_coord_agent_attn
tmux kill-session -t train_rotate_to_attend
tmux kill-session -t train_sparge_agent_attn

# 或者查看所有会话
tmux ls
```

### 4. 数据加载失败

**症状**: `FileNotFoundError` 或 `bad_images.txt` 中有大量图像

**解决方案**:
1. 检查 `IMAGE_ROOT` 路径是否正确
2. 检查标注文件中的路径是否相对于 `IMAGE_ROOT`
3. 验证图像和标签文件是否存在

```bash
# 测试路径
ls -la /your/IMAGE_ROOT/Images/ | head
ls -la /your/IMAGE_ROOT/Labels/ | head
```

### 5. 权限问题

**症状**: `Permission denied`

**解决方案**:
```bash
# 给脚本添加执行权限
chmod +x run_*.sh

# 检查输出目录权限
mkdir -p output logs_tmux runs_tmux
chmod -R 755 output logs_tmux runs_tmux
```

---

## 训练参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_classes` | 分割类别数 | 4 |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 16/32 |
| `--lr` | 学习率 | 0.025 |
| `--momentum` | SGD 动量 | 0.9 |
| `--weight_decay` | 权重衰减 | 1e-4 |
| `--pretrained` | 使用预训练权重 | True |
| `--skip_broken` | 跳过损坏图像 | True |

### Coordinate Attention 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_coord_attn` | 启用坐标注意力 | True |
| `--coord_reduction` | 通道缩减比例 | 32 |

### Triplet Attention 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_triplet_attn` | 启用三维旋转注意力 | True |
| `--triplet_kernel_size` | 卷积核大小 | 7 |
| `--attn_order` | 注意力顺序 | triplet_then_agent |

### Sparge Attention 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_sparge_attn` | 启用稀疏注意力 | True |
| `--sparge_topk` | 稀疏性参数 | 64 |

### Agent Attention 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_agent_attn` | 启用 Agent 注意力 | True |
| `--agent_tokens` | Agent token 数量 | 16 |
| `--agent_heads` | 多头注意力数量 | 8 |
| `--use_agent_bias` | 使用位置偏置 | True |
| `--agent_bias_max_hw` | 最大位置编码尺寸 | 32 |

---

## 性能优化建议

1. **使用多 GPU**：修改脚本添加 `DataParallel` 或 `DistributedDataParallel`
2. **增加 num_workers**：`NUM_WORKERS=4` (根据 CPU 核心数调整)
3. **使用混合精度训练**：添加 `torch.cuda.amp` 支持
4. **数据增强**：在训练脚本中添加更多数据增强策略
5. **学习率调度**：使用 Cosine Annealing 或 Step Decay

---

## 引用

如果使用本项目，请引用相关论文：

```bibtex
@article{coordinate_attention,
  title={Coordinate Attention for Efficient Mobile Network Design},
  author={Hou, Qibin and Zhou, Daquan and Feng, Jiashi},
  journal={CVPR},
  year={2021}
}

@article{triplet_attention,
  title={Rotate to Attend: Convolutional Triplet Attention Module},
  author={Misra, Diganta and Nalamada, Trikay and Arasanipalai, Ajay Uppili and Hou, Qibin},
  journal={WACV},
  year={2021}
}
```

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

如有问题，请提交 GitHub Issue 或联系项目维护者。

**最后更新**: 2026-02-07
