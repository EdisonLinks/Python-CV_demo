# 基于MobileFaceNet和MTCNN的人脸识别系统 Face Recognition System: MobileFaceNet + ArcFace + MTCNN

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.0-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

这是一个基于 **MobileFaceNet** 轻量级网络和 **ArcFace** 损失函数的高效人脸识别系统。项目集成了人脸检测、特征提取、检索比对以及可视化展示等完整流程，支持摄像头实时识别与 Web 端数据展示。

## ✨ 核心特性

- **轻量高效**：采用 MobileFaceNet 作为骨干网络，适合在 CPU 或移动端设备运行。
- **高精度**：使用 ArcFace (Additive Angular Margin Loss) 训练，在 LFW 数据集上表现优异。
- **全流程**：包含 MTCNN 人脸检测、关键点对齐、特征提取与余弦相似度比对。
- **中文支持**：底库图片支持中文文件名（如 `张三_学生.jpg`），识别结果可直接在画面上显示中文姓名。
- **可视化**：提供基于 ECharts 的 Web 可视化界面，展示训练 Loss 与识别统计。

---
## 📖 目录 (Table of Contents)
- [核心架构](#-核心架构)
    - [1. 人脸检测：MTCNN](#1-人脸检测mtcnn)
    - [2. 特征提取：MobileFaceNet](#2-特征提取mobilefacenet)
    - [3. 损失函数：ArcFace](#3-损失函数arcface)
    - [4. 相似度度量](#4-相似度度量)
- [项目结构](#-项目结构)
- [环境准备](#-环境准备)
- [数据准备](#-数据准备)
- [推理流程](#-推理流程)
- [性能指标](#-性能指标)

---

## 🧠 核心架构

本项目的人脸识别流水线包含三个关键阶段：

### 1. 人脸检测：MTCNN
[cite_start]在将图像送入识别网络前，首先使用 **MTCNN (Multi-task Cascaded CNN)** 进行预处理 [cite: 7, 55]。
* **功能**：在图像中精确定位人脸，并回归出 5 个关键特征点（双眼、鼻尖、嘴角）。
* **对齐 (Alignment)**：根据关键点进行仿射变换，将人脸矫正并裁剪为 **112×112** 的标准输入图像。
* [cite_start]**归一化**：像素值归一化至 `[-1, 1]` 区间 ( `(pixel - 127.5) / 128` ) [cite: 55]。

### 2. 特征提取：MobileFaceNet
[cite_start]MobileFaceNet 是针对 MobileNetV2 的改进版，专为人脸识别设计 [cite: 36, 41]。
* **改进核心**：**全局可分离卷积 (GDConv)** 替代了全局平均池化 (GAP)。
    * [cite_start]*原因*：GAP 会丢失空间信息，认为人脸中心和边缘同等重要，这不符合人脸结构特征。GDConv 能够感知不同位置的重要性 [cite: 42, 57, 59]。
* [cite_start]**输出**：将人脸映射到 **128维** 的欧氏空间特征向量 [cite: 65, 73]。

### 3. 损失函数：ArcFace
[cite_start]为了解决传统 Softmax Loss 在人脸验证中区分度不足的问题，本项目采用 ArcFace [cite: 92, 93]。
* **原理**：在特征向量与权重的角度空间中引入 **角度间隔惩罚 (Additive Angular Margin, $m$)**。
* [cite_start]**效果**：迫使同类特征在超球面上更加聚拢（类内紧凑），不同类特征分得更开（类间差异大） [cite: 94, 102]。
* **公式**：
    $$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos \theta_j}}$$

### 4. 相似度度量
[cite_start]使用 **余弦相似度 (Cosine Similarity)** 计算两个 128维 特征向量之间的距离 [cite: 112]。
* **判定**：
    * 值越接近 **1**：表示两人越相似（同一个人）。
    * 值越接近 **-1**：表示截然不同。
    * [cite_start]通常设定一个阈值（如 0.6），超过阈值即认定为同一身份 [cite: 135, 136]。

---

## 🏗️ 项目结构 (Project Structure)

```text
Python-CV_demo/
├── .gitignore                    # Git 忽略配置文件 (忽略了大模型和虚拟环境)
├── requirements.txt              # 项目依赖包列表
├── face_recognition/             # 项目核心目录
│   ├── app/
│   │   ├── detection/            # 【MTCNN】人脸检测模块
│   │   │   ├── face_detect.py
│   │   │   └── utils.py
│   │   ├── models/               # 【模型定义】网络结构
│   │   │   ├── mobilefacenet.py  # 特征提取网络
│   │   │   └── arcmargin.py      # ArcFace Loss
│   │   ├── face_db/              # 【人脸底库】(放入已知身份图片)
│   │   │   ├── Edison.jpg
│   │   │   └── 张三_学生.jpg      # 支持中文文件名
│   │   ├── save_model/           # 【权重文件】(需手动下载放置)
│   │   │   └── mobilefacenet.pth
│   │   ├── utils/                # 工具类 (图像处理、中文绘制等)
│   │   ├── infer.py              # 单张图片推理脚本
│   │   ├── infer_camera.py       # 摄像头实时识别脚本
│   │   ├── train.py              # 模型训练脚本
│   │   └── create_dataset.py     # 数据集制作脚本
│   └── web/                      # 【可视化】Web 展示模块
│       ├── visualization.html    # ECharts 展示页面
│       ├── app.js                # 前端逻辑
│       └── echarts.min.js
└── README.md                     # 项目说明文档
```
## 📥 模型下载 (Model Zoo)

⚠️ **注意**：由于 GitHub 文件大小限制，模型权重文件未上传。请下载下方网盘中的 `face_recoginition_models` 文件夹，并按路径放置。

| 模型文件 (Model) | 说明 (Description) | 本地存放路径 (Local Path) | 下载链接 (Download) |
| :--- | :--- | :--- | :--- |
| **mobilefacenet.pth** | 人脸识别核心权重 | `face_recognition/app/save_model/` | <a href="https://pan.baidu.com/s/17jZt7Ck2vu65vcU2rZC3Iw?pwd=h4bc" target="_blank"><strong>百度网盘下载</strong></a><br>提取码: `h4bc` |
| **mtcnn/*.pth** | 人脸检测权重 (P/R/O Net) | `face_recognition/app/save_model/mtcnn/` | 同上 (包含在链接中) |

> **安装提示**：下载后请确保 `save_model` 目录下的结构与上述路径一致，否则程序会报错找不到模型。
