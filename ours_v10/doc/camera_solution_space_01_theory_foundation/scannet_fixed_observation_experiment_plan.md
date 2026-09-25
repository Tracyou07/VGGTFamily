# Camera Solution Space 01：ScanNet 固定观测验证实施计划

> 日期：2026-08-23
> 分支：`codex/camera_solution_space_01_theory_foundation`
> 基线提交：`8d2036faa9e9f54c6bee6889f563ef54f3b890fb`
> 状态：实施前冻结协议；尚无真实解空间实验结论
> 执行平台：H20；正式数值实验不设 CPU smoke 门

## 0. 本计划解决什么

当前研究已经从“长序列和短序列谁更稳”转向一个更基础的问题：

> 在图像、帧顺序、预处理、相机内参、深度、匹配、评价能量和阈值都固定后，
> 去除全局坐标 gauge 的相机轨迹可接受集合，是否存在远距离可行候选、局部弱
> 可辨识方向，以及候选之间是否能由低能连续路径连接？

本计划不再把不同长度的输入直接互相比较。每个实验实例只使用一个封存后的
8 帧 ScanNet RGB-D observation。后续 VGGT proposal、几何 proposal、局部
Hessian、perturb-refit、continuation 和路径搜索，必须读取同一个不可变 manifest。

不同帧数若以后需要，只能进入独立的 robustness appendix。它不能作为连续冗余、
离散分支存在或分支消失的证据。

## 1. 当前分支和 H20 的实际状态

### 1.1 已经存在的内容

- 代码仓库：`/home/ubuntu/yjh/vggt`
- 当前分支：`codex/camera_solution_space_01_theory_foundation`
- 当前 HEAD：`8d2036faa9e9f54c6bee6889f563ef54f3b890fb`
- 理论文档：
  `doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.md`
- 同一理论文档的 LaTeX 与 PDF 已存在；理论文档明确声明尚无真实实验结论。
- VGGT 权重：
  `/data/yjh/share/pretrained/VGGT-1B/model.safetensors`
- 权重 SHA-256：
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- 权重与当前模型严格核对：1797 个 key 全匹配，无 missing、unexpected 或 shape
  mismatch；全部 tensor 为 F32。
- 可复用运行环境：`/home/ubuntu/anaconda3/envs/vggt-gx/bin/python`
  （Python 3.10.20、PyTorch 2.11.0+cu128、OpenCV 4.9、SciPy 1.15.3）。

### 1.2 ScanNet 下载状态与固定目录

FastVGGT 官方固定 50 场景来自
[`eval/scannet_50.yaml`](https://raw.githubusercontent.com/mystorm16/FastVGGT/main/eval/scannet_50.yaml)。
FastVGGT 的评估入口读取处理后的 `color/`、`pose/` 和 GT PLY；参见
[`eval/eval_scannet.py`](https://raw.githubusercontent.com/mystorm16/FastVGGT/main/eval/eval_scannet.py)。

本项目下载官方 50 个 `.sens` 以及对应 50 个 `_vh_clean_2.ply`，不经过按键确认。
下载采用官方 HTTPS、可续传 partial、Content-Length 验证、完成后原子改名。
50 个 `.sens` 的官方长度合计为 37,267,218,065 bytes（34.708 GiB）。

固定落盘位置：

```text
/data/yjh/share/datasets/ScanNet/
  fastvggt_scannet50.txt
  SOURCES.json
  raw_sens/scans/<scene>/<scene>.sens
  raw/scans/<scene>/<scene>_vh_clean_2.ply
```

下载及输出必须分离：

```text
代码：/home/ubuntu/yjh/vggt
数据：/data/yjh/share/datasets
权重：/data/yjh/share/pretrained
派生 observation 与实验结果：/data/output/camera_solution_space_01
```

`.sens` 的官方 version、header、RGB-D frame record 和压缩定义以
[`ScanNet/SensReader/python/SensorData.py`](https://raw.githubusercontent.com/ScanNet/ScanNet/master/SensReader/python/SensorData.py)
为格式依据。旧 reader 会把整条序列全部载入内存，本计划只借鉴格式，不直接复用
其全量加载/全量导出 API。

### 1.3 当前缺口

当前分支还没有：

- ScanNet `.sens` 随机索引器；
- 固定 observation plan、封存和完整性验证器；
- 与 VGGT 输出独立的 RGB-D 能量；
- gauge-fix、轨迹距离和合法 `SE(3)` 插值；
- 多起点候选 registry；
- Hessian/profile continuation；
- string/NEB 路径搜索与结果卡。

所以现在不能说 ScanNet/VGGT 已经证明连续冗余或离散分支。下载完成只代表数据
准备完成，不代表科学结论成立。

## 2. 研究对象与允许的结论

### 2.1 主研究对象

每个实例使用固定的 8 帧 RGB-D observation，已知度量内参。令首帧规范化后的
轨迹为

$$
q=(P_1,\ldots,P_7)\in SE(3)^7,\qquad P_0=I.
$$

冻结独立能量 $E_{\mathrm{RGBD}}$ 与绝对阈值 $h$ 后，研究

$$
\mathcal S_h=\{q\in SE(3)^7:E_{\mathrm{RGBD}}(q)\le h\}.
$$

主问题拆成三个互不替代的证据对象：

1. **远距离可行候选**：是否存在 $q^A,q^B\in\mathcal S_h$ 且
   $d(q^A,q^B)\ge\delta$？
2. **局部连续弱方向**：是否有稳定小曲率方向，并能通过 profile continuation
   构造一段完整低能路径？
3. **候选间连接/能垒**：远距离候选之间是否能找到全程低于 $h$ 的合法
   `SE(3)` 路径？

### 2.2 允许报告

- 找到两个稳健可行且相距至少 $\delta$ 的候选，可以反证该 surrogate 下的
  $\delta$-近似唯一性。
- 稳定 Hessian 弱方向加约束 profile continuation，可以支持“该固定实例在
  $h$ 尺度存在数值验证的局部低能连续段”。
- 找到并加密验证一条全程低能路径，可以支持“这对候选位于同一经验道路区域”。

### 2.3 禁止报告

- 没找到第二个候选，不能证明唯一。
- NEB/string 搜索失败，不能证明两个分量断开。
- 多个优化 basin、VGGT 的四次 camera refinement、UMAP 空隙或 seed 聚类，
  不能证明离散分支。
- 本 RGB-D surrogate 下的多解，不能直接等同于纯 RGB posterior 的多模态。
- 单个 scene 的结果不能推广成 ScanNet 全体结论，也不能直接推出必须使用 DP/FM。

本阶段没有可信全局下界 $L^*$，因此不认证 $\beta_0>1$，也不把当前最好能量
当作 $E^*$。

## 3. M0：冻结官方 50 场景和数据分割

### 3.1 官方 50 场景

仓库必须保存一份逐行精确等于 FastVGGT 官方 YAML 的配置：

```text
configs/camera_solution_space_01/fastvggt_scannet50.txt
```

下载后的 `/data/.../fastvggt_scannet50.txt` 与仓库配置必须逐字节或逐行规范化后
一致；任何缺失、重复、额外或格式错误 scene 都阻断实验。

### 3.2 事前固定 calibration/evaluation split

不得按看过实验结果后的难易程度挑 scene。使用固定 namespace
`camera_solution_space_01:v1:<scene_id>` 计算 SHA-256，按 digest、scene ID
升序排列，前 12 个作为 calibration，其余 38 个作为 evaluation。

冻结的 12 个 calibration scenes 为：

```text
scene0395_00
scene0466_01
scene0593_00
scene0084_01
scene0631_01
scene0606_01
scene0619_00
scene0071_00
scene0056_00
scene0571_00
scene0177_01
scene0409_01
```

配置写入：

```text
configs/camera_solution_space_01/scannet50_split_v1.json
```

该 split 只用于阈值/数值余量校准与最终 evaluation 隔离，不继承旧分支的实验
结论。

## 4. M0：为每个 scene 封存一个固定 8 帧 observation

### 4.1 两阶段创建，后续不可重采样

Observation 创建分成不可混淆的两个阶段：

1. `index/plan`：只读 `.sens` header 和 frame offsets，生成
   `observation_plan.json`；此阶段确定并保存最终 8 个原始 frame ID。
2. `extract/seal`：只接受该 plan，随机 seek/decode 8 帧，写 RGB、depth、pose、
   intrinsics、匹配 cache 和逐文件 SHA-256，最后原子写入 complete/manifest。

后续实验 CLI 只能接受：

```bash
--observation-manifest /data/output/camera_solution_space_01/observations/<id>/observation_manifest.json
```

不得接受 raw `.sens`、`--frame-count`、重新 sampling 的 seed，或在坏帧时自动换帧。

### 4.2 8 帧固定选择规则 v1

候选序列为

$$
[f,f+15,f+30,\ldots,f+105].
$$

按 $f$ 从小到大选择第一个通过全部门槛的窗口：

- 每帧 RGB 坐标下的有效深度覆盖率至少 55%；
- 每个相邻边至少 150 个冻结 RGB-D 有效匹配；
- 固定 15 条图边中至少 12 条有 100 个以上冻结有效匹配；
- 仅用于窗口筛选和 audit、绝不进入主能量的 sensor GT：相邻平移
  0.04–0.25 m、相邻旋转不超过 8 度、首尾平移 0.30–1.50 m。

固定边集为

$$
\mathcal E=\{(i,i+1)\}_{i=0}^{6}
\cup\{(i,i+2)\}_{i=0}^{5}
\cup\{(0,4),(3,7)\}.
$$

若 scene 没有窗口通过，记录结构化 rejection；不能事后放宽门槛或换一套选择规则。
如果 v1 大面积失败，应在看到候选/路径结果前建立全新的 `selection_version=v2`，
重新生成全部 manifest，并把 v1 失败率完整保留。

### 4.3 Observation manifest 必须封存的内容

```json
{
  "schema": "camera_solution_space_observation/v1",
  "observation_id": "sha256:...",
  "tool_commit": "...",
  "scene_id": "sceneXXXX_YY",
  "split": "calibration|evaluation",
  "sens": {"path": "...", "size": 0, "sha256": "..."},
  "ply": {"path": "...", "size": 0, "sha256": "..."},
  "frame_indices": [0, 15, 30, 45, 60, 75, 90, 105],
  "timestamps_color_us": [],
  "timestamps_depth_us": [],
  "selection_version": "fixed8_stride15_v1",
  "ordered_model_inputs": [],
  "rgb_decode_resize": "native -> long-side 640, bilinear, no crop",
  "depth_to_rgb": "calibrated reprojection, nearest-z z-buffer",
  "depth_valid_m": [0.25, 5.0],
  "edges": [],
  "matcher_config_sha256": "...",
  "artifact_merkle_sha256": "..."
}
```

真实 frame ID 必须来自 `.sens`，上面的数字只是 schema 示例，不能预写进结果。
所有导出文件记录相对路径、size 与 SHA-256。`observation_id` 必须由规范化 manifest
主体和 artifact hashes 生成。

### 4.4 派生输出结构

```text
/data/output/camera_solution_space_01/
  plans/<scene>.observation_plan.json
  observations/<observation-id>/
    observation_manifest.json
    complete.json
    color/<frame-id>.jpg
    depth/<frame-id>.png
    pose_audit/<frame-id>.txt
    intrinsic/
      intrinsic_color.txt
      extrinsic_color.txt
      intrinsic_depth.txt
      extrinsic_depth.txt
    matches/<edge>.npz
```

`pose_audit/` 和 PLY 只能用于筛选、校准或 post-hoc audit。独立目标函数的 Python
接口不能接收 sensor pose、PLY、VGGT depth、VGGT point map 或 VGGT confidence。

## 5. M1：去 gauge 的状态、距离和独立 RGB-D 能量

### 5.1 Gauge fix 与表示

若 world-to-camera 矩阵为 $W_i$，统一规范化为

$$
P_i=W_iW_0^{-1},\qquad P_0=I.
$$

$P_i$ 把首帧相机坐标变换到第 $i$ 帧相机坐标。规范化后不再额外做
Procrustes/SE(3) 对齐。四元数 $r$ 与 $-r$ 必须映射成同一个旋转；禁止对 VGGT
原始 9D pose encoding 做聚类、距离或线性插值。

### 5.2 冻结匹配

每条边的匹配只依赖封存 RGB：

- OpenCV SIFT：`nfeatures=4096`、`contrastThreshold=0.04`、
  `edgeThreshold=10`、`sigma=1.6`；
- mutual 2-NN + Lowe ratio 0.75；
- `cv.setRNGSeed(0)`；MAGSAC fundamental matrix，1 px、confidence 0.999、
  最多 10,000 iterations；
- 两端必须有有效 RGB-aligned depth；
- 按固定 key 排序，冻结到 cache 并散列。

若 OpenCV/build 差异导致 matcher cache hash 改变，必须生成新 observation/objective
版本，不能在原结果上继续。

### 5.3 能量

对匹配 $m=(u_{im},u_{jm})$：

$$
X_i(u)=D_i^{rgb}(u)K_i^{-1}\widetilde u,
\qquad Y_{ijm}=P_jP_i^{-1}X_i(u_{im}).
$$

定义无量纲 3D 与 2D residual：

$$
r_3=\operatorname{diag}(1/0.03,1/0.03,1/0.05)
(Y_{ijm}-X_j(u_{jm})),
$$

$$
r_2=\frac{\pi(K_jY_{ijm})-u_{jm}}{2\ \mathrm{px}}.
$$

Huber 函数

$$
\phi_c(a)=
\begin{cases}
\frac12a^2,&a\le c,\\
c(a-\frac12c),&a>c,
\end{cases}\qquad c=2.5.
$$

固定 pair-balanced 能量

$$
E(q)=\frac1{|\mathcal E|}\sum_{(i,j)\in\mathcal E}
\frac1{|M_{ij}|}\sum_{m\in M_{ij}}
\left[0.7\phi_{2.5}(\|r_3\|)+0.3\phi_{2.5}(\|r_2\|)\right].
$$

若 $Y_z\le0.10$ m，该匹配直接记固定惩罚 10，防止把点投到相机背后逃避残差。

### 5.4 轨迹距离与远距离阈值

$$
d(q,q')=\sqrt{\frac1{7}\sum_{i=1}^{7}
\left[
\frac{\|v_i\|^2}{(0.10\ \mathrm m)^2}
+\frac{\|\omega_i\|^2}{(5^\circ)^2}
\right]},
$$

其中 $(v_i,\omega_i)=\log(P'_iP_i^{-1})$。冻结 $\delta=1$。去重阈值固定为
$d<0.10$。

## 6. M2：事前校准 $m_{\mathrm{eval}}$、$h$ 和判别能力

必须在查看 38 个 evaluation scene 的 VGGT proposal、候选和路径结果前完成并
冻结 calibration card。

### 6.1 数值评价余量

对 12 个 calibration observation 的 sensor GT 轨迹与每个实例 32 个固定 Sobol
扰动，在不同累计顺序/线程设置下各重复 20 次：

$$
m_{\mathrm{eval}}=max\left(10^{-6},
5\max|E_{\mathrm{repeat}}(q)-E_{\mathrm{ref}}(q)|\right).
$$

### 6.2 绝对阈值

$$
h=Q_{0.99}\left(\{E(q_s^{gt})\}_{s=1}^{12}\right)+2m_{\mathrm{eval}}.
$$

稳健可行：$E\le h-2m_{\mathrm{eval}}$。
灰区：$[h-2m_{\mathrm{eval}},h+2m_{\mathrm{eval}}]$。
稳健拒绝：$E>h+2m_{\mathrm{eval}}$。

### 6.3 负控制门

每个 calibration observation 生成 32 个非 gauge Sobol 扰动，并缩放到
$d(q^{bad},q^{gt})=\delta$。至少 95% 必须满足
$E(q^{bad})>h+2m_{\mathrm{eval}}$。

同一 seed 的重复 refit 必须满足

$$
Q_{0.99}[d(q_{refit},q_{refit}^{ref})]<0.10.
$$

任一门失败即停止真实解空间搜索；只能修改 objective/manifest 版本后重新完整校准，
不得事后放宽 $h$。

## 7. M3：解析、合成和表示控制

在真实 ScanNet 结论前必须通过：

| 控制 | 期望 |
|---|---|
| 全局 $SE(3)$ gauge copy | gauge-fix 后完全相同，误差 `<1e-12` |
| 四元数 $r/-r$ | 相同旋转与距离 0 |
| $E(x)=x^2$ | exact minimum 唯一；正容差集为连续厚集合 |
| $E(x,y)=x^2$ | 检出零曲率方向和低能 continuation |
| $(x^2-1)^2,h=0.25$ | 两端间解析 minimax 高度 1 |
| 混合控制 | 两个分量，每个分量内又有连续方向 |
| 合成 RGB-D 唯一 | 扰动后 refit 回归真解，`d<0.05` |
| 合成 RGB-D 连续 | 找到预期旋转自由度与 profile path |
| 20% 匹配/深度污染 | 不得稳健通过 $h$ |

这些控制验证测量工具，不把解析双井当作真实 ScanNet 断开证据。

## 8. M4：候选生成、统一 refit 与 registry

### 8.1 VGGT 只做一次确定性 proposal

严格离线加载：

```python
config = json.load(open("/data/yjh/share/pretrained/VGGT-1B/config.json"))
model = VGGT(**config)
state = safetensors.torch.load_file(
    "/data/yjh/share/pretrained/VGGT-1B/model.safetensors", device="cpu"
)
model.load_state_dict(state, strict=True)
```

`pose_enc_list` 是同一次 forward 的四次 refinement，不作为四个样本。只使用最终
`pose_enc`，经 `pose_encoding_to_extri_intri` 转为矩阵、统一 convention、gauge-fix
后得到 $q_{VGGT}$。ScanNet 已知内参固定，VGGT 预测 FoV 不进入独立目标。

第一轮固定 8 帧、camera-only proposal；VGGT proposal 保存后，后续几何优化不再
重复运行完整模型。

### 8.2 与模型独立的 pose-graph proposal

利用冻结的 RGB-D 3D–3D matches：每边以 3 点 Umeyama RANSAC（5 cm、最多
10,000 次）估相对 `SE(3)`，再作鲁棒 pose-graph 初始化，得到 $q_{PG}$。

### 8.3 多起点

- $q_{VGGT}$ 与 $q_{PG}$；
- 对二者各加 64 个固定 Sobol 42 维扰动；
- RMS 半径循环 0.5、1、2；
- continuation 发现的端点也进入同一 registry。

所有 seed 使用完全相同的 float64 Lie-group robust refit，最多 300 次迭代；停止
条件为归一化梯度 `L_inf < 1e-6`，且连续 5 次相对能量变化 `<1e-9`。

Registry 必须记录 observation/objective/calibration/checkpoint/code hashes、seed、
起点来源、收敛状态、逐边能量、总能量、pairwise distance matrix 和去重映射。
GT 只进入 post-hoc audit，不参与候选筛选。

## 9. M5：局部弱方向和 profile continuation

在稳健 interior 候选 $q_*$ 处使用无量纲局部坐标：

$$
P_i(x)=\exp([0.0872665x_i^\omega,0.10x_i^v]^\wedge)P_i^*.
$$

冻结当前 IRLS 活动权重，float64 autograd 计算 residual Jacobian $J$ 与
$H_{GN}=J^TJ$。用中心差分步长 $10^{-3}$ 复核，并同时检查
$5\times10^{-4}$、$2\times10^{-3}$。小特征值跨步长变化超过 10% 时只标记
数值不稳定，不作物理解读。

取 $a_w=0.5$，候选弱方向阈值

$$
\tau(q_*)=\frac{2[h-E(q_*)-2m_{\mathrm{eval}}]_+}{a_w^2}.
$$

对每个 $\lambda_k\le\tau$ 的方向做约束 profile refit：

$$
\min_qE(q)\quad\mathrm{s.t.}\quad v_k^Tx=a,
\quad a\in\{\pm0.25,\pm0.5,\pm0.75,\pm1\}.
$$

约束残差必须 `<1e-2`。相邻 profile 点用合法 `SE(3)` geodesic 加密；整段均
满足 $E\le h-2m_{\mathrm{eval}}$ 时，才能报告“数值验证的低能连续段”。小
Hessian 特征值本身不等于连续冗余证据。

## 10. M6：远距离候选的合法路径和数值能垒

对每个稳健可行且 $d(q^A,q^B)\ge\delta$ 的候选对，先按能量排序，pilot 最多
搜索 10 对。初始合法路径为 32 个区间的产品 `SE(3)` geodesic：

$$
P_i^{(\ell)}=
\exp\left(\frac{\ell}{32}\log(P_i^B(P_i^A)^{-1})\right)P_i^A.
$$

随后运行 Lie-group string/NEB，最多 2000 步。最终递归细分到每段长度不超过
$0.05\delta$，并检查中点、四分点。报告上界

$$
H_U^{num}(A,B)=\max_{path\ samples}[E(q)+m_{\mathrm{eval}}].
$$

- 若 $H_U^{num}\le h-2m_{\mathrm{eval}}$，支持存在数值验证的低能连接路径。
- 若预算内找不到，只能报告“搜索预算内未找到连接”。
- 没有经过验证的 minimax 下界 $H_L>h+m_{\mathrm{eval}}$，不得写“离散分支已证实”。

## 11. 代码与测试布局

计划新增：

```text
configs/camera_solution_space_01/
  fastvggt_scannet50.txt
  scannet50_split_v1.json
  selection_fixed8_stride15_v1.json
  matcher_sift_magsac_v1.json
  objective_rgbd_v1.json

pre_experiments/camera_solution_space_01/
  __init__.py
  contracts.py
  sens_index.py
  observation.py
  matching.py
  se3.py
  trajectory.py
  rgbd_energy.py
  calibration.py
  controls.py
  vggt_proposal.py
  pose_graph_proposal.py
  refit.py
  registry.py
  local_geometry.py
  continuation.py
  path_search.py
  reporting.py

scripts/camera_solution_space_01/
  preflight_scannet50.py
  plan_observations.py
  seal_observations.py
  validate_observations.py
  calibrate_objective.py
  run_controls.py
  run_candidates.py
  run_local_geometry.py
  run_path_search.py
  build_report.py

tests/camera_solution_space_01/
  test_contracts.py
  test_sens_index.py
  test_observation.py
  test_matching.py
  test_se3.py
  test_trajectory.py
  test_rgbd_energy.py
  test_calibration.py
  test_controls.py
  test_registry.py
  test_local_geometry.py
  test_continuation.py
  test_path_search.py
  test_reporting.py
```

真实数据、权重、cache、结果绝不提交 Git。

## 12. 三人分工与递进交付

三个人不是各跑一套互不兼容的实验，而是通过冻结接口串起来。每个人可以先在
解析/合成 fixture 上并行开发；真实 ScanNet 运行按 Gate 顺序推进。

### Part A / 人员 1：数据、observation 与 objective contract

负责文件：

```text
configs/camera_solution_space_01/*
pre_experiments/camera_solution_space_01/contracts.py
pre_experiments/camera_solution_space_01/sens_index.py
pre_experiments/camera_solution_space_01/observation.py
pre_experiments/camera_solution_space_01/matching.py
scripts/camera_solution_space_01/{preflight,plan,seal,validate}_*.py
tests/.../{test_contracts,test_sens_index,test_observation,test_matching}.py
```

递进任务：

1. 先写 synthetic `.sens` fixture 和失败测试；验证 index 阶段不解码 payload，
   能记录 color/depth offset、size、pose、timestamp。
2. 实现官方 header/version/compression、frame 边界和有限 pose 校验。
3. 实现 fixed8 plan；冻结真实 frame ID，不在 seal 阶段重选。
4. 只随机读取 8 帧；深度保存为原生 `uint16` PNG，RGB/depth/intrinsics 尺寸一致。
5. 实现 depth-to-RGB z-buffer、冻结 SIFT/MAGSAC cache、逐文件 hash 与 Merkle root。
6. 真实运行前计算容量预算，并确保源数据目录前后 fingerprint 不变。

Part A 的交付门：一个 calibration scene 的 immutable observation 通过 deep validate，
所有 hashes 可重复，候选 runner 只靠 manifest 能加载完整观测。

### Part B / 人员 2：几何状态、独立能量、校准和候选

负责文件：

```text
pre_experiments/camera_solution_space_01/{se3,trajectory,rgbd_energy}.py
pre_experiments/camera_solution_space_01/{calibration,controls}.py
pre_experiments/camera_solution_space_01/{vggt_proposal,pose_graph_proposal}.py
pre_experiments/camera_solution_space_01/{refit,registry}.py
scripts/camera_solution_space_01/{calibrate_objective,run_controls,run_candidates}.py
对应 tests
```

递进任务：

1. 先写 `SE(3)` exp/log、组合、逆、gauge-copy、四元数符号与 geodesic 测试。
2. 实现 frozen-match RGB-D energy，并用合成唯一/连续/污染场景验证。
3. 完成 12-scene calibration；在任何 evaluation 候选可见前冻结 calibration card。
4. 严格加载 VGGT 权重，只将最终 camera 输出当一个 proposal。
5. 实现独立 pose-graph proposal、Sobol 多起点、统一 refit 和候选 registry。
6. 输出远距离稳健候选对和审计用 GT error，但不作拓扑结论。

Part B 的交付门：M2/M3 全部通过；registry 包含完整 hashes、能量、距离、收敛状态，
并证明 `pose_enc_list` 未被当成 samples。

### Part C / 人员 3：局部连续性、路径搜索与证据报告

负责文件：

```text
pre_experiments/camera_solution_space_01/{local_geometry,continuation}.py
pre_experiments/camera_solution_space_01/path_search.py
pre_experiments/camera_solution_space_01/reporting.py
scripts/camera_solution_space_01/{run_local_geometry,run_path_search,build_report}.py
对应 tests
```

递进任务：

1. 先在解析唯一、连续 fiber、双井和混合模型上通过 Hessian/profile/path tests。
2. 实现 autograd Jacobian/GN Hessian 与三步长有限差分一致性检查。
3. 实现约束 profile continuation；对每段做合法 `SE(3)` 加密验证。
4. 实现 Lie-group string/NEB；输出数值路径上界而不是伪造断开证书。
5. 对 Part B 的远候选最多先跑 10 对，并记录失败预算、终止原因和能量曲线。
6. 生成逐 observation result card 和 38-scene aggregate；自动限制结论用语。

Part C 的交付门：报告清楚区分“找到低能连接”“预算内未找到”“没有远候选”三种
状态，不把后两种写成离散分支证据。

### 12.4 集成责任

- Part A 冻结 `observation_manifest` schema 后，Part B/C 只读，不私自扩展。
- Part B 冻结 `objective_card`、`calibration_card` 和候选 registry schema 后，
  Part C 只读。
- schema 改动必须递增版本并使旧 artifact 明确失效，不能静默兼容。
- 三个人的正式命令都在 H20 执行，结果统一写 `/data/output/camera_solution_space_01`。

## 13. TDD 与验证门

每个功能遵循：先写会失败的最小测试，确认 RED；再写最小实现，确认 GREEN；最后
做局部重构。小型解析/合成单元测试可以使用 CPU，但它们是正确性测试，不是正式
CPU smoke 或研究结论。所有真实 ScanNet/VGGT 数值实验在 H20 上运行。

基础验证：

```bash
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest \
  discover -s tests/camera_solution_space_01 -v

/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m py_compile \
  pre_experiments/camera_solution_space_01/*.py \
  scripts/camera_solution_space_01/*.py
```

真实数据 Gate：

```bash
python scripts/camera_solution_space_01/preflight_scannet50.py \
  --dataset-root /data/yjh/share/datasets/ScanNet \
  --output-root /data/output/camera_solution_space_01 \
  --strict --dry-run

python scripts/camera_solution_space_01/plan_observations.py \
  --dataset-root /data/yjh/share/datasets/ScanNet \
  --split calibration --selection-version fixed8_stride15_v1

python scripts/camera_solution_space_01/seal_observations.py \
  --plans-root /data/output/camera_solution_space_01/plans \
  --split calibration

python scripts/camera_solution_space_01/validate_observations.py \
  --root /data/output/camera_solution_space_01/observations \
  --split calibration --deep
```

正式 GPU 运行前重新检查 `nvidia-smi`。当前优先候选 GPU 为 5，其次 3；明确避开
GPU 4/6/7，且不自动挑卡。命令必须显式：

```bash
CUDA_VISIBLE_DEVICES=5 \
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python \
scripts/camera_solution_space_01/run_candidates.py \
  --observation-manifest <sealed-manifest> \
  --checkpoint /data/yjh/share/pretrained/VGGT-1B/model.safetensors
```

GPU 选择只是当前快照，正式启动前必须重新确认进程归属和空闲显存。第一轮使用
8 帧 camera-only proposal，并至少保留 30–40 GiB 可用显存。

## 14. Go / No-Go 顺序

```text
G0  50 个 .sens + 50 个 PLY 下载、数量和长度验证
 ↓
G1  synthetic .sens 索引/随机解码/manifest tests 全过
 ↓
G2  12 个 calibration observation 封存并 deep validate
 ↓
G3  m_eval、h、delta 与负控制通过并冻结 calibration card
 ↓
G4  解析/合成/表示控制全过
 ↓
G5  evaluation pilot：1 scene 的 VGGT/PG/Sobol/refit registry
 ↓
G6  pilot 的 Hessian + profile continuation
 ↓
G7  若有远候选，运行最多 10 对 path search；否则明确记录“未发现”
 ↓
G8  扩展到 38 个 evaluation scenes 并汇总
```

任一 Gate 失败必须停在该层修复和重新版本化。不能绕过校准/控制，直接对 50 个
scene 跑大规模候选后再根据结果调阈值。

## 15. 最终结果卡

每个 observation 至少输出：

```text
observation / objective / calibration / checkpoint / code hashes
h, m_eval, delta, calibration pass/fail
U = minimum found energy（不声称等于 E*）
candidate registry 与 pairwise distance matrix
Hessian spectrum、步长稳定性、profile continuation 结果
searched far pairs 与 H_U^num matrix
GT/PLY post-hoc audit
所有失败预算和停止原因
```

最终汇总按以下决策表写结论：

| 观察 | 可写结论 | 不可写结论 |
|---|---|---|
| 稳健远候选存在 | 反证 $\delta$-近似唯一 | 已有离散分支 |
| 稳定弱方向 + 完整 profile path | 支持局部低能连续段 | 存在 exact 连续 fiber |
| 远候选间找到完整低能路径 | 候选对经验连通 | 整个子水平集只有一个分量 |
| 路径搜索失败 | 预算内未找到连接 | 已证明断开 |
| 没有远候选 | 当前搜索未反证近似唯一 | 已证明唯一 |

## 16. 近期最小下一步

1. 等待 G0 下载完成并做 100 个对象的最终数量/长度核对。
2. 按 Part A 的前四个测试先实现 `.sens` header/frame-offset 随机索引。
3. 仅对一个 calibration scene 生成真实 8 帧 plan、seal、deep validation。
4. 通过后扩到 12 个 calibration scenes，冻结 $m_{\mathrm{eval}}$ 与 $h$。
5. 在看 evaluation 结果前完成全部解析/合成控制。
6. 只在这些门都通过后启动第一个正式 H20 evaluation pilot。

这条顺序保证最终回答的是“同一个固定观测下有没有多解/弱方向/低能连接”，而
不是再次回答“换了序列长度后输出是否变化”。
