# Assumptions and Open Questions

本文档记录第一轮骨架搭建中所有**未确认**信息，避免伪造字段。

## A. 数据字段（最高优先级待确认）

### A1. 多视图/多帧图像
- TODO: 单个样本是否同时支持多视图与时间维？
- TODO: 图像文件组织方式（按 identity / sequence / frame）未确认。
- TODO: 图像分辨率是否固定，是否允许每相机不同分辨率。

### A2. 相机参数
- TODO: 内参字段命名（`K` or `intrinsics`）未确认。
- TODO: 外参格式（`[R|t]` / `cam2world` / `world2cam`）未确认。
- TODO: 坐标系约定（OpenCV / OpenGL）未确认。
- TODO: 畸变参数是否提供、是否必须使用未确认。

### A3. FLAME 跟踪输出
- TODO: FLAME参数字段命名（shape/expression/pose）未确认。
- TODO: 是否包含每帧 head pose 与 global transform 未确认。
- TODO: FLAME 模板版本（拓扑、UV layout）未确认。

## B. Stage 1 相关协议
- TODO: DINOv2 选型（ViT-S/B/L）与输出层策略未确认。
- TODO: 图像特征投影到 mesh 的采样方式（rasterize / nearest / barycentric）未确认。
- TODO: canonical UV 分辨率（如 256/512/1024）未确认。
- TODO: 多视图 UV 融合策略（平均/置信加权/可学习）未确认。
- TODO: UV refinement 网络输入输出通道协议未确认。
- TODO: position/normal map 的归一化范围未确认。
- TODO: mesh refinement 监督信号与损失权重未确认。

## C. Stage 2 相关协议
- TODO: Gaussian anchor 初始化来源优先级（UV vs refined mesh）未确认。
- TODO: 每个 anchor 的属性集合（scale/rotation/opacity/color/features）未确认。
- TODO: differentiable renderer 接口（第三方库 / 自研）未确认。
- TODO: 是否需要 temporal consistency loss 未确认。

## D. 训练与评估
- TODO: stage1/stage2 是否分离训练还是端到端微调未确认。
- TODO: 批组织方式（按身份、按帧、按视图）未确认。
- TODO: 评估指标（PSNR/SSIM/LPIPS/几何误差）主次优先级未确认。
- TODO: train/val/test 划分协议未确认。

## E. 暂不纳入第一版（已确认）
- diffusion
- VGGT / FastVGGT
- hair
- relighting
- prompt-conditioned decoder（仅保留扩展接口）
- residual off-surface branch

## F. 与代码 TODO 对应
- `datasets/base_dataset.py`: 样本字典字段协议 TODO
- `models/geometry/flame_adapter.py`: FLAME 输入字段/坐标约定 TODO
- `models/stage1_prior/uv_fusion.py`: 融合策略 TODO
- `models/stage2_gaussian/gaussian_decoder.py`: Gaussian 属性集合 TODO
- `models/render/gaussian_renderer.py`: renderer backend 选择 TODO
- `trainers/stage1_trainer.py` / `trainers/stage2_trainer.py`: 损失组合 TODO

