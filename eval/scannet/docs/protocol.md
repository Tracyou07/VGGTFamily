# FastVGGT ScanNet 评测口径

本仓库按用户要求复用现有 FastVGGT 的 ScanNet 评测脚本。兼容口径固定为 `fastvggt_scannet_evo132`，采用官方 evo1.32.0 的对齐语义；原 FastVGGT requirements未锁定evo版本，而1.30、1.32、1.36之后对同时启用对齐及原点对齐的处理不同。当前环境的兼容层复现1.32的先Sim(3)、再原点对齐顺序，独立数值对照覆盖全部原结果字段。原始来源是 H20 上的 `eval/7scenes/reference/FastVGGT-main`，对应项目 https://github.com/mystorm16/FastVGGT 。`reference/FastVGGT/` 保存脚本、辅助函数、场景表、许可证及内容校验和。评测结果中的源代码指纹确定本次运行使用的具体实现。

## 场景与选帧

使用原 `scannet_50.yaml` 的50个场景。缓存从原始 `.sens` 导出全部具有有限位姿的 RGB帧、原始帧号和 c2w TXT，直接使用官方 `_vh_clean_2.ply` 作为 GT几何。数据预处理默认不截断帧数。现有 `.sens` 来自 v1路径、GT网格来自 v2路径；[ScanNet官方获取脚本](https://kaldir.vc.in.tum.de/scannet/download-scannet.py)明确说明 v2继续使用同一份 v1 `.sens`，这组来源符合官方安排。

评测时执行原脚本的 `build_frame_selection`：按有效帧号排序；超过 `input_frame` 时保留首帧，再用 `max(1, (有效帧数-1)//(input_frame-1))` 作为步长选择余下帧，最后截断到上限。这并非包含末帧的 linspace 采样。默认上限1000；较小上限是单独的运行配置，结果记录实际帧号。

## 原脚本的六个结果字段

| 字段 | 单位 | 原脚本计算方式 |
|---|---|---|
| `ate` | m | evo translation APE的RMSE，启用对齐、尺度校正及原点对齐 |
| `are` | 度 | evo rotation-angle APE的RMSE |
| `rpe_trans` | m | 相邻已选帧、delta=1的translation RPE的RMSE |
| `rpe_rot` | 度 | 相邻已选帧、delta=1的rotation-angle RPE的RMSE |
| `chamfer_distance` | m | 下述点云处理后的双向平均距离之和 |
| `inference_time_ms` | ms | 适配器记录的推理耗时；具体计时范围见模型元数据 |

### 位姿约定

原代码先将 GT c2w位姿变换到首个 GT帧坐标系，再取逆得到 w2c；预测外参也以 w2c矩阵交给 `eval_trajectory`。evo在这里读取的是 w2c矩阵的平移及旋转。因此，本仓库报告的是 **FastVGGT参考脚本口径**，不能把这些数值直接称为通常以相机中心 c2w计算的 ATE。适配器统一提供 c2w，在评测边界转换为原脚本需要的表示。

### Chamfer 约定

1. 预测点云通过首个 GT位姿变换到 ScanNet世界坐标系。
2. 按 GT与预测点云包围盒对角线长度之比缩放预测点云，并对齐包围盒中心。这一步独立于轨迹对齐，使用了 GT的几何范围。
3. 每侧超过100000点时，分别用 NumPy随机种子33采样到100000点。
4. 两侧分别按0.05m体素下采样。
5. 求预测→GT及GT→预测的最近邻欧氏距离；每条距离截断到 `chamfer_max_dist`，默认0.5m。
6. 返回两侧平均距离的**和**，不再除以2。因此默认截断下该值的理论上界是1.0m。

原脚本未输出 Acc、Comp、NC或F-score；这六个原字段是本仓库的主结果。不同点云过滤策略、帧数、checkpoint或源代码产生的结果必须分开比较。

## 基础设施修复范围

独立评测入口负责校验数据和模型、选择解释器、加载正确权重格式、调用各模型真实推理接口、保存运行来源并严格汇总。对空输入、缺失帧、非有限数值、非法位姿、缺失 GT及不完整指标报错。已有结果仅在全部来源信息一致时允许恢复，汇总包含本次请求中的已完成场景，且任一场景失败都会明确标为不完整。

FastVGGT默认保留其深度置信度阈值1.0，原脚本自己的点数上限应用于几何对齐之后。Long和SLAM使用各自原生重建过程和显式记录的置信度过滤。小规模GPU测试验证接口连通性；最终ScanNet50成绩需要相同完整配置的正式运行。

FastVGGT的真实GPU回归发现BF16位姿解码可产生约0.008的旋转正交误差。运行副本仅将pose_enc转为FP32后解码相机，并使用同一解码结果反投影深度；模型前向保持BF16。这个数值稳定性修复已单独记录在来源清单和模型元数据，原始脚本快照仍保留。
