# FeedFarwardGenerativeGaussianAvatar

Geometry-Consistent Canonical Priors for Feedforward Generative Gaussian Avatars

## 当前阶段（第二轮）
本轮聚焦于：**真实路径接入 + schema 适配 + 几何基础设施 + 可视化验证脚本**。

> 暂不实现完整 Stage 1 prior builder / Stage 2 Gaussian decoder 训练逻辑。

## 已确认的真实路径（默认写入配置）
- FLAME 资产根目录：`/home/yuanyuhao/VHAP/asset/flame`
- FLAME 模型：`/home/yuanyuhao/VHAP/asset/flame/flame2023.pkl`
- FLAME 模板网格：`/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj`
- FLAME masks：`/home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl`
- UV masks：`/home/yuanyuhao/VHAP/asset/flame/uv_masks.npz`
- raw NeRSemble 根目录：`/home/yuanyuhao/FastAvatar/data/NeRSemble`
- processed FastAvatar-style 根目录（第一优先）：
  `/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full`
- 样例相机目录：
  `/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191`

## 数据与后端策略
- **Primary schema**：processed FastAvatar-style (`transforms.json` + `processed_data/*`)。
- raw NeRSemble：当前仅提供 adapter 占位，不编造未确认字段。
- Gaussian renderer backend：优先 `gsplat`。
- Mesh projection/rasterization backend：优先 `nvdiffrast`，保留 `PyTorch3D` fallback 接口。

## 第一版协议（已落到配置）
- DINOv2: `vitb14`
- UV resolution: `256`
- Stage1 / Stage2: 分开训练
- Stage1 UV fusion: `confidence_weighted_average`
- Stage2: surface Gaussian only
- metrics: `PSNR / SSIM / LPIPS`
- prompt-conditioned decoder: 接口保留，不启用

## 目录结构
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

## 关键配置文件
- `configs/base.yaml`
- `configs/data/nersemble_fastavatar.yaml`
- `configs/model/geometry.yaml`
- `configs/model/stage1_prior.yaml`
- `configs/model/stage2_gaussian.yaml`

如果你的本地路径与默认不同，优先修改：
1. `configs/base.yaml` 中 `paths.*`
2. `configs/data/nersemble_fastavatar.yaml` 中 `root/raw_root/subject_id/sequence_id/camera_id`
3. `configs/model/geometry.yaml` 中 `geometry.flame.*`

## 环境安装
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

## 第二轮验证命令（建议顺序）
### 1) 可视化读取真实样本
```bash
python scripts/visualize_sample.py \
  --camera-dir /home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191 \
  --index 0 \
  --out outputs/visualize/sample_rgb.png
```

### 2) FLAME mesh 投影并保存 overlay + UV 占位结果
```bash
python scripts/debug_mesh_overlay.py \
  --camera-dir /home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191 \
  --index 0 \
  --flame-model /home/yuanyuhao/VHAP/asset/flame/flame2023.pkl \
  --flame-template /home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj \
  --flame-masks /home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl \
  --uv-masks /home/yuanyuhao/VHAP/asset/flame/uv_masks.npz \
  --uv-resolution 256 \
  --transform-mode unknown \
  --out-overlay outputs/debug/mesh_overlay.png \
  --out-uv-mask outputs/debug/uv_valid_mask.npy \
  --out-uv-pos outputs/debug/uv_position_map.npy \
  --out-uv-nrm outputs/debug/uv_normal_map.npy
```

## 说明
- 当前 overlay 的 mesh 来自 head template + FLAME translation（稳态调试优先）。
- 完整 FLAME 形变与可微 rasterization 内核接入放在后续轮次。
- `transform_matrix` 的 cam2world / world2cam 仍需你用可视化结果最终确认（见 `docs/assumptions.md`）。
