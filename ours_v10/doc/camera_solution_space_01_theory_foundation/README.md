# Camera Solution Space 01 — Theory Foundation

本目录是新研究线的理论起点。它继承 `origin/main@15e96cc` 的稳定 VGGT
基线，但不继承旧实验分支的专属提交、结果或结论。

当前状态：**已有理论定义、证据规范和 ScanNet 固定观测实施计划；尚无真实
解空间实验结果，也没有训练结论。**

## 产物

- `camera_trajectory_solution_space.md`：便于协作和评审的 Markdown 主文档。
- `scannet_fixed_observation_experiment_plan.md`：固定 8 帧 RGB-D observation、
  独立能量、控制实验、H20 执行门和三人分工的递进实施计划。
- `camera_trajectory_solution_space.tex`：与 Markdown 同结构的排版源文件。
- `references.bib`：报告引用的原始论文和正式出版物。
- `../../output/pdf/camera_trajectory_solution_space_theory.pdf`：编译后的 PDF。

## 核心问题

在图像、帧顺序、预处理、相机约束和评价目标全部固定后，相机轨迹的
可接受解集合在去除 gauge 后究竟有多大、是否道路连通、候选之间需要跨越
多高的能垒，以及是否存在局部弱可辨识方向？

报告有意不把问题简化成“唯一 / 连续低维 / 离散分支”三选一。正容差
子水平集通常是满维厚集合，而且断开的分量内部仍可能存在连续弱方向。
因此正文采用四量描述：商空间直径、道路分量数、minimax 高度剖面/矩阵和
局部弱可辨识维数。

## 分支边界

```text
origin/main@15e96cc
└── codex/camera_solution_space_01_theory_foundation

旧 014–023 实验分支的专属提交 ── 不在本分支祖先链中
```

本分支保留 VGGT 的现有单次前馈与 Camera Head 迭代细化代码。现有
`pose_enc_list` 是同一次确定性解码中的迭代状态，不是从固定条件解空间
采出的多个 mode。

## 构建

在仓库根目录运行：

当前 Windows MiKTeX 环境未提供 `latexmk` 所需的 Perl，因此使用等价的显式
编译链：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs `
  doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
bibtex tmp/pdfs/camera_trajectory_solution_space
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs `
  doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=tmp/pdfs `
  doc/camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex
Copy-Item tmp/pdfs/camera_trajectory_solution_space.pdf `
  output/pdf/camera_trajectory_solution_space_theory.pdf
```

文档编译和版式检查可在本机完成；后续正式数值实验与训练默认只在 H20
服务器执行，不设计本地 CPU smoke 作为研究门槛。
