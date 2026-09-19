# NRGBD unified evaluation

独立的 NRGBD 评测基建，严格采用 FastVGGT 的 `kf=10` 口径，并让五个模型共用同一数据加载器和评分器：

- Original VGGT
- VGGT-Long
- StreamVGGT
- VGGT-SLAM
- VGGT-Ω

## 固定协议

数据位于 `/data/yjh/share/datasets/NRGBD`。评测九个正式场景，排除 `archives`。每十帧取一帧，输入为 518×392，评分中心裁剪为 224×224；每个点云最多 999,999 点；使用 0.1 m point-to-point ICP。输出 Acc、Comp、NC1、NC2、NC 及其中位数。

原始 FastVGGT 使用文件系统顺序和未固定的随机抽样；本实现固定场景/帧顺序和 seed，并在结果 provenance 中记录这一点。

## 使用

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/nrgbd
export PYTHONPATH=$PWD/src
python -m nrgbd_eval check --config configs/h20.json
python -m nrgbd_eval doctor --config configs/h20.json --model vggt
```

真实运行一次只跑一个模型：

```bash
bash scripts/run_model.sh vggt GPU_ID
bash scripts/run_model.sh vggt_long GPU_ID
bash scripts/run_model.sh streamvggt GPU_ID
bash scripts/run_model.sh vggt_slam GPU_ID
bash scripts/run_model.sh vggt_omega GPU_ID
```

恢复已中断运行：

```bash
bash scripts/run_model.sh MODEL GPU_ID --resume
```

本次开发不启动真实 GPU 评测。先执行 `doctor`；它会如实报告本机缺失的源码或权重。当前 H20 上五个后端的源码、权重、Python 入口和依赖路径均通过 doctor 静态检查；该检查不加载模型权重。真实运行前仍需重新执行，只有通过后才可启动。

## 结构和边界

`nrgbd_eval.data` 独占数据选择和 GT；模型只收到场景 ID、有序帧 ID 和 RGB 绝对路径。各后端在独立 Python 进程中执行并输出统一 NPZ。共享 runner 完成 GT 反投影、FastVGGT 尺度/位移归一、ICP、指标、原子提交和断点恢复。适配器不得读取深度或实现指标。

结果写入 `results/<model>`。每个场景只有写入 `COMPLETE` 后才算完成；`--resume` 只跳过 provenance 完全匹配的场景。缺场景的 summary 明确标记为 `partial`。

推理时间、适配器时间、场景总时间、PyTorch allocated/reserved 峰值和外部 nvidia-smi 采样分别记录。外部显存可能包含同卡其他进程。

## 测试

```bash
bash scripts/verify_cpu.sh
```

CPU 测试只用合成数据和假路径，不读取真实 checkpoint，不分配 CUDA。真实模型 API、GPU 显存、速度和最终指标需在后续授权运行中验证。
