# Assumptions and Open Questions (Round 2)

## 1) 已确认信息（已从 TODO 移除）

### 1.1 Primary data schema
当前第一优先支持：**processed FastAvatar-style schema**

Root 示例：
`/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full/017/EXP-1-head_part-1/cam_220700191`

### 1.2 FLAME 资产路径（已确认）
- `/home/yuanyuhao/VHAP/asset/flame/flame2023.pkl`
- `/home/yuanyuhao/VHAP/asset/flame/head_template_mesh.obj`
- `/home/yuanyuhao/VHAP/asset/flame/FLAME_masks.pkl`
- `/home/yuanyuhao/VHAP/asset/flame/uv_masks.npz`

### 1.3 canonical_flame_param.npz 字段（已确认）
- `translation`: `(1,3)` float32
- `rotation`: `(1,3)` float32
- `neck_pose`: `(1,3)` float32
- `jaw_pose`: `(1,3)` float64
- `eyes_pose`: `(1,6)` float32
- `shape`: `(300,)` float32
- `expr`: `(1,100)` float32

### 1.4 flame_param/*.npz 字段（已确认）
- `translation`: `(1,3)` float32
- `rotation`: `(1,3)` float32
- `neck_pose`: `(1,3)` float32
- `jaw_pose`: `(1,3)` float32
- `eyes_pose`: `(1,6)` float32
- `shape`: `(300,)` float32
- `expr`: `(1,100)` float32

### 1.5 transforms.json 顶层字段（已确认）
- `frames`
- `cx`
- `cy`
- `fl_x`
- `fl_y`
- `h`
- `w`
- `camera_angle_x`
- `camera_angle_y`
- `timestep_indices`
- `camera_indices`

已知样例统计：
- `len(frames) = 162`
- `len(camera_indices) = 1`

### 1.6 transforms.json frame 字段（已确认）
- `timestep_index`
- `timestep_index_original`
- `timestep_id`
- `camera_index`
- `camera_id`
- `cx`
- `cy`
- `fl_x`
- `fl_y`
- `h`
- `w`
- `camera_angle_x`
- `camera_angle_y`
- `transform_matrix`
- `file_path`
- `fg_mask_path`
- `flame_param_path`

### 1.7 processed_data/<timestep_id> 内容（已确认）
- `bg_color.npy`
- `intrs.npy`
- `landmark2d.npz`
- `mask.npy`
- `rgb.npy`

### 1.8 后端偏好（已确认）
- Gaussian renderer: `gsplat`
- Mesh projection/rasterization: `nvdiffrast`
- fallback interface: `PyTorch3D`

### 1.9 第一版训练协议（已确认）
- DINOv2: `vitb14`
- UV resolution: `256`
- Stage1 / Stage2 分开训练
- Stage1 UV fusion: confidence-weighted average
- Stage2: surface Gaussian only
- metrics: PSNR / SSIM / LPIPS
- prompt-conditioned decoder: 接口保留，不启用

## 2) 暂未完全确认（仍需人工确认）
- `transform_matrix` 语义：`cam2world` 还是 `world2cam`。
- FLAME rotation / jaw / neck / eyes 的具体旋转参数化约定（axis-angle / 其他）及坐标系方向。
- `uv_masks.npz` 内部键名语义（不同 mask 的角色划分）。
- raw NeRSemble 全量 schema（目前仅确认 camera_params 有 subject 分目录形式）。

## 3) 当前实现边界（本轮）
- 已实现：真实路径配置、processed schema 数据读取、基础 mesh overlay、UV 占位输出。
- 未实现：完整 FLAME 前向形变、真实可微 rasterization kernel、Stage1/Stage2 主体训练逻辑。

## 4) 与代码 TODO 对应
- `datasets/nersemble_dataset.py`: raw NeRSemble adapter 待补全。
- `models/geometry/flame_wrapper.py`: full FLAME deformation 待集成。
- `models/geometry/uv_ops.py`: 使用真实 `uv_masks.npz` 语义替换占位 mask。
- `models/geometry/camera_utils.py`: transform mode 自动判别仍不可靠，需人工标定验证。
