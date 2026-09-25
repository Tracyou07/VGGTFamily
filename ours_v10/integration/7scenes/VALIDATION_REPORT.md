# ours_v4 7Scenes 开发验收报告

实测实现提交：`cd0abd132b094ba06ba3d68f59d7d406f3e8c775`。
起始提交：`a3b13e561de97b0699e42f1cd29fe0918b5483ce`。
后续提交仅增加本报告与采样清单，实测运行源码哈希与交付源码一致。

## 验收结论

- CPU：ours_v4 与新增集成测试 **24/24**，仓库其他测试 **8/8**，共 **32/32**。
- 指标回归：原样抽取现有 Long adapter 的 evaluator；使用真实 criterion 和 Open3D，固定 OMP_NUM_THREADS=1 后四项指标逐项完全相同。
- 数据清单：kf=3 为18序列/5678帧；kf=10 为18序列/1700帧。另实际调用原 SevenScenes loader 检查 chess/seq-03 的 kf=3 全部334帧，路径、顺序与518×392尺寸一致。
- GPU 等价：chess/seq-03，kf=10 的55帧，窗口30/30/15；实际 batch=1 对比 batch=2，局部输出与最终拼接结果均通过。
- 门禁没有 GT 对齐：局部最大中心差1.91e-7，拼接后2.46e-7（原始预测坐标单位）；最大局部深度差4.06e-6；尾窗口逐元素相同。
- 预声明容差：atol=.02、rtol=.02、中心距离<=.01、旋转<=.5度；未根据结果调整。
- 完整序列 smoke：只跑 chess/seq-03、kf=10、100帧，不截断。成功。
- 实际 resume：输出 SKIP verified，仍只有1个attempt、1条序列记录，没有重新推理。
- 只标记 SMOKE_COMPLETE；summary.valid_sequences=1，complete=false，没有伪造18/18的COMPLETE。
- 忙GPU检查实际拒绝了GPU0；非法kf=7被launcher拒绝。
- 再次核验了封存哈希、部署副本、100个唯一frame ID、first-window ownership与合法旋转。

## Smoke结果

| 指标 | 数值 |
|---|---:|
| Acc (m) | 0.016623253 |
| Comp (m) | 0.018052596 |
| NC1 | 0.631157913 |
| NC2 | 0.634973018 |
| Mean NC | 0.633065465 |
| 预处理 (s) | 1.6176 |
| VGGT前向 (s) | 15.5479 |
| overlap拼接 (s) | 1.0329 |
| 重建合计 (s) | 19.5669 |
| 协议评测，另计 (s) | 130.4468 |
| CUDA allocated (GiB) | 39.9606 |
| CUDA reserved (GiB) | 63.1934 |
| 序列RSS，100ms采样 (GiB) | 4.8563 |

重建合计另含打包/传输、预测转换与point-map反投影，不含诊断导出、GT读取和协议评测。
CPU进程累计高水位另存在metrics.json，不能与本序列RSS混淆。
GPU：H20物理4号，UUID `GPU-83919d3d-cd82-2f1d-fbca-da6d44a8c78d`。
环境：`{"Pillow": "11.3.0", "numpy": "1.26.4", "open3d": "0.19.0", "opencv-python": "4.9.0.80", "safetensors": "0.8.0", "torch": "2.11.0+cu128"}`。

## 结果位置

- 门禁：`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/results/ours_v4_kf10_20260921T024855447318360Z_3909807`
- 门禁报告：`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/results/ours_v4_kf10_20260921T024855447318360Z_3909807/gate_report.json`
- Smoke：`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/results/ours_v4_kf10_20260921T025042854723872Z_3914601`
- 单序列完整产物：`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/results/ours_v4_kf10_20260921T025042854723872Z_3914601/sequences/chess__seq-03/attempt_0001`
- CPU报告：`/home/ubuntu/yjh/feedforwardreconstruct/ours_v4/integration/7scenes/CPU_TEST_REPORT.txt`
- 18序列双kf清单：`/home/ubuntu/yjh/feedforwardreconstruct/ours_v4/integration/7scenes/SPLIT_AUDIT.json`
- 使用说明：`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/README_OURS_V4.md`

全部大产物保留在H20；已有ScanNet结果、旧模型adapter和历史目录未改动。

## 正式命令（本次未执行）

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes
bash run_ours_v4_job.sh GPU_ID 3
bash run_ours_v4_job.sh GPU_ID 10
```

将GPU_ID替换为空闲物理GPU编号。同一GPU串行运行。每次使用新时间戳目录。
恢复命令、门禁与smoke命令见README。完整18序列结果尚未验证，单序列通过不等于全部18序列已通过。
ours_v4使用depth-unprojection point map，VGGT-Long使用point-head world_points；不能称为纯batching消融。
测试出现既有evo ResourceWarning，GPU运行出现原模型autocast弃用提示，未影响测试或结果。

## 所有新增文件

版本管理于ours_v4；未修改起始版本已有文件：

- experiments/sevenscenes/common.py
- experiments/sevenscenes/engine.py
- experiments/sevenscenes/protocol.py
- experiments/sevenscenes/runner.py
- integration/7scenes/adapters/eval_ours_v4_7scenes.py
- integration/7scenes/run_ours_v4_job.sh
- integration/7scenes/README_OURS_V4.md
- integration/7scenes/CPU_TEST_REPORT.txt
- integration/7scenes/SPLIT_AUDIT.json
- integration/7scenes/VALIDATION_REPORT.md
- tests/ours_v4/test_sevenscenes.py
- tests/ours_v4/test_seven_protocol.py
- tests/ours_v4/test_seven_engine.py

部署到eval/7scenes的新增文件（与版本管理副本SHA256一致）：

- adapters/eval_ours_v4_7scenes.py
- run_ours_v4_job.sh
- README_OURS_V4.md
