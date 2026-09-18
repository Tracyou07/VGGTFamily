# ScanNet evaluation — FastVGGT reference protocol

独立仓库位于 H20：`/home/ubuntu/yjh/feedforwardreconstruct/eval/scannet`。

按要求复用 FastVGGT 的 ScanNet 评测脚本、选帧方式和指标计算；外围增加原始数据准备、七种模型的真实推理接口、环境检查、来源记录和严格断点恢复。原始脚本及许可证保存在 `reference/FastVGGT/`，运行时使用的兼容副本位于 `scannet_eval/`。

## 当前验证范围

- 原 FastVGGT ScanNet50 场景表：50/50 场景，83,632 个有效 RGB/位姿对已导出并通过完整内容校验；GT 网格共有 7,859,543 个有限顶点。
- 七种后端都已完成同一真实场景的 8 帧 GPU 推理和参考指标计算。Long 使用两个重叠分块，SLAM 使用两个子图。
- 数据、指标兼容性、后端、命令入口、失败处理和断点恢复由仓库内测试覆盖。
- **没有运行七种模型 × ScanNet50 的完整成绩评测。** 8 帧测试证明接口可用，不能代替完整成绩或证明 1,000 帧的显存需求。短序列测试也不证明长序列闭环效果。

## H20 环境与数据

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/scannet
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
$PY -m scannet_eval doctor --config configs/h20.json
```

`doctor` 不分配 GPU，逐个使用配置中的解释器检查后端。SLAM 自动使用 `monst3r`；其他模型使用 `vggt-gx`。所有模型源码和权重都引用 H20 上的现有共享路径。`configs/h20.json` 是实际运行配置。

| 后端名 | 实现 | 推理特点 |
|---|---|---|
| `vggt_original` | 原版 VGGT | VGGT 原始推理接口 |
| `vggt_star` | 当前 VGGT* | 当前仓库的模型接口 |
| `fastvggt` | FastVGGT | 原预处理、merging 和置信度设置 |
| `streamvggt` | StreamVGGT | 原生 KV cache 推理 |
| `long` | Long | 原生重叠分块、Sim(3) 拼接及检索 |
| `slam` | SLAM | 原生子图、SALAD 检索及图优化 |
| `omega` | Omega | 原生 512/balanced 输入和模型解码 |

准备好的缓存：`/data/yjh/share/datasets/ScanNet/prepared_scannet50_v1`。

```text
<prepared_root>/<scene>/
  color/000000.jpg
  pose/000000.txt
  calibration/
  gt/<scene>_vh_clean_2.ply
  manifest.json
```

RGB 和位姿保留原始帧号。准备阶段保留全部有效帧，评测时只执行一次 FastVGGT 原选帧算法；不会先截断缓存再采样。本评测直接使用官方 PLY 作为 GT 几何，不使用此前 7Scenes 的 `.depth.proj.png` 链接。

```bash
# 完整校验，默认校验内容哈希
$PY -m scannet_eval verify --config configs/h20.json

# 需要另建缓存时指定新的输出目录；源数据重叠检查在写入前完成
$PY -m scannet_eval prepare --config configs/h20.json \
  --prepared-root /data/yjh/share/datasets/ScanNet/prepared_scannet50_new
```

不要把准备输出指向原始扫描目录。当前已完成的缓存不需要再次生成。Long 的附加依赖默认安装到本仓库 `.runtime/long_deps`，这是运行时生成目录，可通过 `scripts/setup_long_deps.sh` 重建，不升级共享环境。

## 复用 FastVGGT 入口

先检查 H20 当前 GPU、现有进程和磁盘空间，并指定有足够余量的 GPU。下例选择物理 GPU 0，输出必须使用新的目录。

```bash
scripts/preflight_h20.sh 0 20000
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY eval_scannet.py \
  --config configs/h20.json \
  --scenes scene0150_00 \
  --input_frame 8 \
  --merging 0 --merge_ratio 0.9 --depth_conf_thresh 1.0 \
  --output_path results/my_fastvggt_smoke
```

输出继承 FastVGGT 布局：

```text
results/my_fastvggt_smoke/input_frame_8/
  run_manifest.json
  scene0150_00/metrics.json
  scene0150_00/result.json
  average_metrics.json
  all_scenes_metrics.json
  summary.json
  failures/                 # 失败时记录
```

`--data_dir` 可指向符合本仓库 manifest 契约的准备缓存；`--gt_ply_dir` 如另行指定，必须与已验证 GT 的内容哈希一致。支持 `--ckpt_path`、`--chamfer_max_dist`、`--plot true`。未知参数或不支持的可视化参数会报错，不会静默忽略。

移除 `--scenes` 即请求配置中的全部 50 个场景；正式运行请明确指定所需 `--input_frame`，默认值为 1000。大场景推理前需要独立评估显存容量。

## 其他模型与恢复

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m scannet_eval run --config configs/h20.json \
  --model streamvggt --scenes scene0150_00 --max-frames 8 \
  --output results/my_streamvggt_smoke

# 使用完全相同的命令和输出目录，加 --resume
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m scannet_eval run --config configs/h20.json \
  --model streamvggt --scenes scene0150_00 --max-frames 8 \
  --output results/my_streamvggt_smoke --resume

$PY -m scannet_eval aggregate --output results/my_streamvggt_smoke
```

每种模型在独立解释器进程中执行，避免多个 `vggt` 命名空间冲突。模型参数可修改配置文件后写入新的输出目录。所有实际参数、checkpoint 哈希、模型及评测代码指纹、解释器版本、数据 manifest 和选中帧号都会保存。

恢复必须匹配原运行来源；代码、配置、权重、选帧或环境变化会拒绝复用旧结果。每个场景的 `metrics.json` 与 `result.json` 都必须存在，完整保留 13 个有限数值且相互一致；文件缺失、损坏或不一致时会拒绝恢复和汇总。已经完成的场景纳入汇总，失败或缺失场景使 `summary.complete=false` 且命令返回非零；不能只看平均值文件判断评测完成。零成功场景不会生成伪造的全零成绩。

## 原脚本的指标语义

协议标识：`fastvggt_scannet_evo132`。汇总保留六个主字段：`chamfer_distance`、`ate`、`are`、`rpe_rot`、`rpe_trans`、`inference_time_ms`；单场景保留原脚本的全部 13 个字段。

原脚本将归一化 GT 与预测的 **w2c** 矩阵交给 evo，几何单独按 GT 包围盒缩放和居中。Chamfer 为双向均值的和，距离默认截断到 0.5 m；原脚本不计算 Acc、Comp、NC。请参阅 [完整口径](docs/protocol.md)，不要将这些结果混称为另一套标准 c2w/统一 Sim(3) 评测。

两项数值兼容改动均已公开记录：

1. 原 FastVGGT 未锁定 evo 版本。本仓库固定为官方 evo 1.32.0 的先 Sim(3)、再原点对齐语义；对照原版函数验证全部指标，兼容 H20 已安装的较新 evo。
2. FastVGGT 的 BF16 位姿编码转为 FP32 后再解码相机和反投影深度，修复真实测试中约 0.008 的旋转正交误差；模型前向继续使用 BF16。

## 代码验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m unittest discover -s tests -v
```

原生 SLAM 的专用检查在 `monst3r` 环境执行。本仓库没有替换原始数据、历史结果或各模型源码，也没有向外部 Git 服务推送。
