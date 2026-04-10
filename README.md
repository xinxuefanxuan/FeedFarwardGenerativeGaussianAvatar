# FeedFarwardGenerativeGaussianAvatar

Geometry-Consistent Canonical Priors for Feedforward Generative Gaussian Avatars

## 1. 项目目标（第一轮）
本仓库当前版本是**研究骨架初始化版本**，目标是：
- 提供可维护、可扩展的 PyTorch 项目结构。
- 为后续两阶段方法实现（Stage 1 canonical prior / Stage 2 Gaussian avatar）打好接口。
- 明确记录不确定数据字段与待确认事项，避免“伪实现”。

> 当前**不实现完整模型**，仅提供最小可运行入口与模块签名。

## 2. 方法概述（研究计划）
### 输入
- 稀疏多视图或稀疏多帧人头图像
- 相机参数
- FLAME 跟踪输出

### Stage 1: geometry-consistent canonical prior
1. 图像编码器（默认预留 DINOv2 冻结接口）
2. 图像特征投影到粗 FLAME mesh
3. 映射到 canonical UV 空间
4. 多视图 UV 融合
5. UV refinement
6. UV position/normal maps
7. mesh refinement

### Stage 2: feedforward Gaussian avatar generation
1. 从 UV / refined mesh 初始化 Gaussian anchors
2. 解码 Gaussian 属性
3. differentiable Gaussian rendering

## 3. 本轮明确不做
- diffusion
- VGGT / FastVGGT
- hair
- relighting
- prompt-conditioned decoder（仅保留接口）
- residual off-surface branch

## 4. 目录结构
```text
configs/
data/
assets/
datasets/
models/
  encoders/
  geometry/
  stage1_prior/
  stage2_gaussian/
  render/
trainers/
evaluation/
scripts/
utils/
notebooks/
docs/
```

## 5. 环境安装
### Conda（推荐）
```bash
conda env create -f environment.yml
conda activate ff-gga
```

### Pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. 快速开始（骨架运行）
```bash
python scripts/train_stage1.py --config configs/base.yaml
python scripts/train_stage2.py --config configs/base.yaml
python scripts/prepare_dataset.py --config configs/base.yaml
python scripts/evaluate.py --config configs/base.yaml
```

这些脚本当前只做配置加载与模块连通性验证，用于后续增量研发。

## 7. 配置系统说明
- `configs/base.yaml`：公共默认项与 stage 选择。
- `configs/data/*.yaml`：数据源类型/路径/字段占位。
- `configs/model/*.yaml`：Stage 1/2 模块开关与基础超参。
- `configs/train/*.yaml`：训练策略与优化器占位。

## 8. 待确认事项
请优先查看：
- `docs/assumptions.md`

其中列出了所有未确认字段、命名与协议；代码中的 `TODO` 与该文档一一对应。

## 9. 当前状态总结
### 已实现（骨架）
- 标准目录结构
- 环境配置文件
- 配置 YAML 模板
- 最小可运行脚本入口
- Stage 1 / Stage 2 关键模块类签名
- 假设与风险文档

### 尚未实现（占位）
- 具体网络结构与训练逻辑
- 几何投影、UV 融合、mesh refinement 细节
- Gaussian renderer 的真实可微实现
- 数据集真实解析与字段映射

