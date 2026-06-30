# 数据说明 Data Guide

## 📊 数据集概述

本项目支持两个骨架动作识别数据集：

### 1. NTU RGB+D Dataset

**官方网站**：http://rose1.ntu.edu.sg/datasets/actionrecognition.asp

**数据集信息**：
- **动作类别数**：60类
- **骨架关节数**：25个关节点
- **数据格式**：3D骨架坐标 (x, y, z)
- **数据形状**：`(N, C, T, V, M)`
  - N：样本数量
  - C：坐标维度 (3: x, y, z)
  - T：时间帧数 (300帧)
  - V：关节点数 (25个)
  - M：人数 (最多2人)
- **大小**：5.8GB (仅3D骨架数据)

**动作类别**：
```
1-10:   drink, eat, brush teeth, brush hair, drop, pickup, throw, sit down, stand up, clap
11-20:  read, write, tear up paper, wear jacket, take off jacket, wear shoe, take off shoe, wear glasses, take off glasses, put on hat
21-30:  take off hat, cheer up, wave hand, kick, reach into pocket, hop, jump up, phone call, play with phone, type on keyboard
31-40:  point to something, take selfie, check time, rub hands together, nod head/bow, shake head, wipe face, salute, put palms together, cross hands
41-50:  sneeze/cough, stagger, fall, touch head, touch chest, touch back, touch neck, nausea, use fan, punch/slap
51-60:  kick other, push other, pat on back, point finger at other, hug, give something, touch pocket, handshake, walk towards, walk apart
```

**评估协议**：
- **Cross-Subject (xsub)**：训练集和测试集使用不同的人
- **Cross-View (xview)**：训练集和测试集使用不同的视角

### 2. Kinetics-Skeleton Dataset

**数据集信息**：
- **动作类别数**：400类
- **骨架关节数**：18个关节点 (OpenPose格式)
- **数据格式**：2D/3D骨架坐标
- **大小**：7.5GB

## 📥 数据下载

### 方案1：下载预处理数据（推荐）

从Google Drive下载已预处理的数据：

**NTU RGB+D**：
```bash
# 下载链接
https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb

# 解压到项目目录
cd 2s_st-gcn
unzip st-gcn-processed-data.zip
```

**Kinetics-Skeleton**：
```bash
# 下载链接
https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ

# 或百度网盘
https://pan.baidu.com/s/1dwKG2TLvG-R1qeIiE4MjeA
```

### 方案2：从原始数据生成

#### NTU RGB+D

```bash
# 1. 从官网下载原始骨架数据
# http://rose1.ntu.edu.sg/datasets/actionrecognition.asp

# 2. 解压数据
unzip nturgbd_skeletons_s001_to_s017.zip

# 3. 生成预处理数据
python tools/ntu_gendata.py --data_path <path-to-nturgbd+d_skeletons>
```

#### Kinetics-Skeleton

```bash
# 1. 下载Kinetics骨架数据
# https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ

# 2. 生成预处理数据
python tools/kinetics_gendata.py --data_path <path-to-kinetics-skeleton>
```

## 📁 数据目录结构

正确的数据目录结构：

```
2s_st-gcn/
├── data/
│   ├── ntu/
│   │   ├── xsub/
│   │   │   ├── train_data.npy          # 训练数据 (N, C, T, V, M)
│   │   │   ├── train_label.pkl         # 训练标签
│   │   │   ├── val_data.npy            # 验证数据
│   │   │   └── val_label.pkl           # 验证标签
│   │   └── xview/
│   │       ├── train_data.npy
│   │       ├── train_label.pkl
│   │       ├── val_data.npy
│   │       └── val_label.pkl
│   └── kinetics/
│       ├── train_data.npy
│       ├── train_label.pkl
│       ├── val_data.npy
│       └── val_label.pkl
```

## 🔍 数据格式说明

### .npy 文件格式

骨架数据存储为numpy数组：

```python
import numpy as np

# 加载数据
data = np.load('data/ntu/xsub/train_data.npy')

# 数据形状: (N, C, T, V, M)
print(data.shape)  # 例如: (40320, 3, 300, 25, 2)

# 参数说明:
# N = 40320  # 训练样本数
# C = 3      # 坐标维度 (x, y, z)
# T = 300    # 时间帧数
# V = 25     # 关节点数 (NTU RGB+D)
# M = 2      # 最多2人
```

### .pkl 文件格式

标签数据存储为pickle文件：

```python
import pickle

# 加载标签
with open('data/ntu/xsub/train_label.pkl', 'rb') as f:
    sample_name, label = pickle.load(f)

# sample_name: 样本文件名列表
# label: 样本标签列表 (0-59)

print(f"样本数: {len(sample_name)}")
print(f"第一个样本: {sample_name[0]}, 标签: {label[0]}")
```

## 🧪 数据验证

### 检查数据是否存在

```bash
python -c "
import os
import numpy as np

data_path = 'data/ntu/xsub/train_data.npy'

if os.path.exists(data_path):
    data = np.load(data_path, mmap_mode='r')
    print(f'✓ Data found: {data_path}')
    print(f'  Shape: {data.shape}')
    print(f'  Dtype: {data.dtype}')
    print(f'  Size: {data.nbytes / (1024**3):.2f} GB')
else:
    print(f'✗ Data not found: {data_path}')
    print('  Please download data from:')
    print('  https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb')
"
```

### 可视化骨架数据

使用提供的可视化工具：

```bash
# 从.npy文件加载并可视化
python tools/visualize_skeleton.py \
    --data data/ntu/xsub/train_data.npy \
    --skeleton-type ntu-rgb+d \
    --output-dir outputs/visualizations \
    --frame-interval 10 \
    --max-frames 20

# 创建视频动画（需要ffmpeg）
python tools/visualize_skeleton.py \
    --data data/ntu/xsub/train_data.npy \
    --skeleton-type ntu-rgb+d \
    --output-dir outputs/visualizations \
    --create-video \
    --fps 30
```

## 🎯 测试数据（无需下载）

如果没有真实数据，可以使用生成的测试数据：

### 方案1：使用假模型生成测试数据

```bash
# 生成随机骨架数据用于测试
python -c "
import numpy as np
import os

# 创建测试数据
N = 10  # 10个样本
C = 3   # x, y, z坐标
T = 300 # 300帧
V = 25  # 25个关节
M = 2   # 2人

test_data = np.random.randn(N, C, T, V, M).astype(np.float32)

# 保存
os.makedirs('data/test', exist_ok=True)
np.save('data/test/test_data.npy', test_data)

print(f'✓ Generated test data: data/test/test_data.npy')
print(f'  Shape: {test_data.shape}')
"
```

### 方案2：使用compare_inference.py自动生成

`tools/compare_inference.py` 会自动生成随机数据如果没有提供真实数据：

```bash
# 不提供--data-path参数，自动生成随机数据
python tools/compare_inference.py \
    --weights fake_models/test.pt \
    --openvino-xml export_models/test.xml \
    --num-samples 10 \
    --device CPU
```

## 📊 数据统计信息

### NTU RGB+D 60 Classes

| 拆分 | 样本数 | 训练集 | 测试集 |
|------|--------|--------|--------|
| **Cross-Subject (xsub)** | 56,880 | 40,320 | 16,560 |
| **Cross-View (xview)** | 56,880 | 37,920 | 18,960 |

### 骨架关节点定义 (NTU RGB+D 25 joints)

```
0:  Base of spine        13: Left knee
1:  Middle of spine      14: Left ankle
2:  Neck                 15: Left foot
3:  Head                 16: Right hip
4:  Left shoulder        17: Right knee
5:  Left elbow           18: Right ankle
6:  Left wrist           19: Right foot
7:  Left hand            20: Spine (shoulder)
8:  Right shoulder       21: Left hand tip
9:  Right elbow          22: Left thumb
10: Right wrist          23: Right hand tip
11: Right hand           24: Right thumb
12: Left hip
```

## 🚀 快速开始

### 1. 使用真实数据（如果已下载）

```bash
# 训练模型
python main.py recognition \
    -c config/st_gcn.twostream/ntu-xsub/train.yaml \
    --work_dir ./work_dir

# 测试模型
python main.py recognition \
    -c config/st_gcn.twostream/ntu-xsub/test.yaml \
    --weights work_dir/xxx.pt
```

### 2. 使用假数据测试OpenVINO流程

```bash
# 1. 创建假模型
python tools/create_fake_model.py \
    --output fake_models/test.pt \
    --num-class 60

# 2. 导出到OpenVINO
python tools/export_to_openvino.py \
    --weights fake_models/test.pt \
    --output-dir export_models

# 3. 比较推理性能（自动生成随机数据）
python tools/compare_inference.py \
    --weights fake_models/test.pt \
    --openvino-xml export_models/test.xml \
    --num-samples 10 \
    --device CPU

# 4. 可视化结果
python tools/visualize_skeleton.py \
    --data data/test/test_data.npy \
    --skeleton-type ntu-rgb+d \
    --output-dir outputs/visualizations
```

## 💡 常见问题

### Q1: 为什么我的data目录是空的？

**A**: 数据需要手动下载。请从以下链接下载：
- NTU RGB+D: https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb
- Kinetics: https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ

### Q2: 可以不下载真实数据吗？

**A**: 可以！使用假数据测试OpenVINO集成：
```bash
python tools/create_fake_model.py --output fake_models/test.pt
python tools/export_to_openvino.py --weights fake_models/test.pt --output-dir export_models
python tools/compare_inference.py --weights fake_models/test.pt --openvino-xml export_models/test.xml
```

### Q3: 数据太大，下载很慢怎么办？

**A**: 
1. 使用百度网盘链接（针对中国用户）
2. 只下载NTU RGB+D的xsub分割（~2GB）而非完整数据
3. 使用假数据测试流程，真实训练时再下载

### Q4: 如何查看数据内容？

**A**: 使用可视化工具：
```bash
python tools/visualize_skeleton.py --data <path-to-npy> --skeleton-type ntu-rgb+d --output-dir outputs
```

### Q5: 数据格式不对怎么办？

**A**: 确保数据形状为 `(N, C, T, V, M)`:
```python
import numpy as np
data = np.load('your_data.npy')
print(f"Shape: {data.shape}")  # 应该是 (N, 3, 300, 25, 2)

# 如果shape不对，可能需要转置
if data.shape != (N, 3, 300, 25, 2):
    data = data.transpose(...)  # 根据实际情况调整
```

## 📚 相关资源

- **NTU RGB+D官网**: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
- **Kinetics官网**: https://deepmind.com/research/open-source/kinetics
- **OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **ST-GCN论文**: https://arxiv.org/abs/1801.07455

## 📝 数据使用协议

使用NTU RGB+D和Kinetics数据集时，请遵守相应的使用协议和引用要求。

**NTU RGB+D引用**:
```
@inproceedings{shahroudy2016ntu,
  title={NTU RGB+ D: A large scale dataset for 3D human activity analysis},
  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle={CVPR},
  year={2016}
}
```

---

**最后更新**：2026-06-29  
**数据类型**：骨架序列 (3D坐标)  
**状态**：✅ 完整文档
