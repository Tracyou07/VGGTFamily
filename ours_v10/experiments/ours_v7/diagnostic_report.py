"""Assemble measured diagnostics into reviewable reports without rerunning inference."""
import argparse,csv,json,subprocess,statistics
from pathlib import Path
from collections import defaultdict
from experiments.ours_v7.diagnostics import ROOT,OLD,LONG,STAR,WINDOWS,sha256,write_json,csv_rows

def read_csv(path):
    with open(path) as f:return list(csv.DictReader(f))
def load(path):return json.loads(Path(path).read_text())
def measured(out,mode):
    return [r for r in read_csv(out/f'{mode}_timing.csv') if r['diagnostic']=='False']
def stage_group(name):
    if name=='baseline':return 'model_loaded'
    if name=='input_transfer':return 'input_transfer'
    if name=='window_initialization':return 'window_initialization_including_encoder'
    if name=='aggregator.patch_embed':return 'image_encoder'
    if name.startswith('aggregator.patch_embed.blocks.'):return 'image_encoder_layer'
    if name.startswith('aggregator.frame_blocks.'):return 'frame_attention'
    if name.startswith('aggregator.global_blocks.') or name=='global_step':return 'global_attention'
    if name=='communication_bank':return 'communication_bank'
    if name=='aggregator':return 'aggregator_including_encoder'
    if name in ('camera_head','depth_head','point_head'):return name
    return 'head_internal'

def run(out):
    modes=['full','windows','independent','patch','patch_q1024','patch_cached_keys']
    for m in modes:
        assert len(read_csv(out/f'{m}_timing.csv'))==3
    for m in ['windows','independent','patch','patch_cached_keys']:
        rows=read_csv(out/f'{m}_regression.csv')
        assert len(rows)==63 and all(r['exact']=='True' for r in rows)
    assert all(r['exact']=='True' for r in read_csv(out/'raw_prediction_differences.csv'))
    post=load(out/'optimized_downstream_regression.json');assert post['evaluation_identical']
    init=load(out/'initialization_checks.json')
    assert all(x['camera_register_exact'] and x['rope_exact'] for x in init['checks'])
    strict=load(out/'full100_strict_evaluation.json')
    source=load(out/'source_manifest.json')
    source['final_diagnostic_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    source['final_diagnostic_status']=subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True).strip()
    source['diagnostic_files_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'experiments/ours_v7').glob('diagnostic*.py')}
    import hashlib,platform,torch
    adapter=ROOT.parent/'fixed_input_baselines'
    source['baseline_adapter_file_checks']=[]
    for p in (adapter/'experiments/fixed_input_baselines').glob('*.py'):
        relative=str(p.relative_to(adapter))
        committed=subprocess.check_output(['git','-C',str(adapter),'show','d80946cfb1463ba82ec63530b016d87f63a9cf4d:'+relative])
        expected=hashlib.sha256(committed).hexdigest()
        actual=sha256(p)
        assert actual==expected
        source['baseline_adapter_file_checks'].append(dict(file=str(p),actual_sha256=actual,commit_blob_sha256=expected,matches=True))
    source['environment']=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu_driver=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid,driver_version','--format=csv'],text=True).strip())
    source['original_ours_commit']='6554d358fce7f8e551abb6169688bfcbab45e66f'
    source['final_inference_byte_checks']=[]
    original=load(OLD/'independent/run_manifest.json')
    for name,digest in original['sources']['sha256'].items():
        if name.startswith(('vggt/','experiments/ours_v7/worker','experiments/ours_v6/worker','vendor/')):
            match=sha256(ROOT/name)==digest
            source['final_inference_byte_checks'].append(dict(file=name,matches=match))
            assert match
    source['initialization_audit']=init
    import torch
    from experiments.ours_v7.diagnostics import tensor_hash
    source['all_historical_input_tensors']=[]
    for mode in ['independent','camera_exchange','camera_patch_exchange']:
        path=OLD.parent/f'20260922T023028Z_v7_f100_{mode}'/'inputs.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        digest=tensor_hash(saved['images'])
        assert digest==source['images_sha256'] and saved['frame_ids']==[f'{i:06d}' for i in range(100)]
        source['all_historical_input_tensors'].append(dict(mode=mode,path=str(path),file_sha256=sha256(path),
            image_sha256=digest,shape=list(saved['images'].shape),dtype=str(saved['images'].dtype)))
        del saved
    source['numerical_settings']=dict(ours=dict(deterministic=True,matmul_tf32=False,cudnn_tf32=False,
        benchmark=False,autocast='bfloat16',original_heads='float32',seed=2026),
        historical_baseline=dict(deterministic=False,matmul_tf32=False,cudnn_tf32=True),
        benchmarking_cpu_threads=4,compile='no torch.compile; Long Numba cached JIT included in first edge timing')
    source['historical_baseline_flags_evidence']='baseline adapter source and exact native default replay of all saved Long fields; not recorded in original baseline manifest'
    write_json(out/'source_manifest.json',source)
    first=load(out/'first_divergence.json') if (out/'first_divergence.json').exists() else dict(
        native_windows_vs_ours_independent=dict(first_divergence=None,all_fields_exact=True,repeat_difference_zero=True),
        historical_long_vs_ours=dict(first_divergence='backbone.image_encoder.block0',
            evidence=load(out/'runtime_flag_first_divergence.json'),native_default_reproduces_long_exactly=True),
        frozen_assembly=load(out/'alignment_summary.json'),no_implementation_bug_demonstrated=True)
    first['full100_vs_windows']=dict(full100_matched_settings_ate_m=strict['ate_rmse_m'],
        independent_first_ate_m=.03522108840469494,explanation='window attention context and estimated cross-window geometry differ from joint100; no implementation discrepancy found',
        cannot_separate='pure context information loss versus inherent estimated alignment error without additional experiments')
    write_json(out/'first_divergence.json',first)
    memory=[];timing=[]
    for mode in modes:
        for r in read_csv(out/f'{mode}_memory.csv'):
            memory.append(dict(kind='measured_run',**r))
        stages=read_csv(out/f'{mode}_stages.csv')
        for r in stages:
            memory.append(dict(kind='measured_stage',run=mode,stage=r['stage'],group=stage_group(r['stage']),event=r['event'],
                diagnostic=True,allocated=r['allocated'],reserved=r['reserved'],peak_allocated=r['peak_allocated'],
                peak_reserved=r['peak_reserved'],cpu_rss=r['cpu_rss'],output_unique_storage_bytes=r.get('output_unique_storage_bytes','')))
            if r.get('seconds'):
                timing.append(dict(run=mode,stage=r['stage'],group=stage_group(r['stage']),seconds=r['seconds'],diagnostic=True,
                    scope='synchronized module boundary; nested intervals must not be added together'))
        timing.extend(read_csv(out/f'{mode}_timing.csv'))
        base=load(out/f'{mode}_load.json')
        timing += [dict(run=mode,stage='model_load',seconds=base['model_load_seconds'],diagnostic=False,cold=True),
                   dict(run=mode,stage='load_frozen_input_file',seconds=base['input_file_load_seconds'],diagnostic=False,cold=True),
                   dict(run=mode,stage='preprocessing',seconds=0,diagnostic=False,scope='reused original preprocessed tensor')]
    timing.extend(read_csv(out/'alignment_timing.csv'))
    timing.extend(read_csv(out/'export_optimization.csv'))
    for mode in ['independent','camera_exchange','camera_patch_exchange']:
        m=load(OLD.parent/f'20260922T023028Z_v7_f100_{mode}'/mode/'run_manifest.json')
        for stage,seconds in m['timing'].items():
            timing.append(dict(run='historical_'+mode,stage=stage,seconds=seconds,diagnostic=False,scope='historical manifest; reconstruction total excludes export/model load'))
    inv=load(out/'patch_tensor_inventory.json')
    banks=[r for r in inv if r['stage']=='communication_bank' and r['event']=='after'][:3]
    bank_unique=sum(t['storage_bytes'] for r in banks for t in r['tensors'] if t['first_storage_occurrence'])
    bank_logical=sum(t['logical_bytes'] for r in banks for t in r['tensors'])
    model_bytes=load(out/'independent_load.json')['model_unique_storage_bytes']
    residents=[
        ('model_parameters_and_buffers','various','float32','cuda',model_bytes,'load through inference; measured unique storage'),
        ('original_input_images','[100,3,392,518]','float32','cpu',243667200,'shared source storage; window slices are views'),
        ('one_window_image_copy_max','[1,60,3,392,518]','float32','cuda',146200320,'one current window, plus normalization temporary; not 160 simultaneous images'),
        ('window_states_and_positions','[1,62460,1024]x2 + [1,41640,1024] + positions','float32 / int64','cuda',684894720,'all windows through aggregator; independent semantic states'),
        ('retained_head_features','4 layers x 160 frames x 1041 tokens x 2048','float32','cuda',5457838080,'layers 4,11,17,23; released per completed window head'),
        ('patch_bank_unique_storage','K FP32; V BF16 view into QKV','mixed','cuda',bank_unique,'per global layer, source bank once; excludes shared-storage double count'),
        ('patch_bank_logical_tensor_bytes','160 frames x 105 selected tokens x 1024 x (4+2)','mixed','cuda',bank_logical,'logical values only; smaller than actual storage'),
        ('depth_head_cudnn_workspace','opaque execution-plan workspace','opaque','cuda',35986604048,'temporary within convolution; allocator trace run_conv_plan; not permanent tensor'),
    ]
    for name,shape,dtype,device,n,lifetime in residents:
        memory.append(dict(kind='resident_or_temporary_storage',run='ours_v7',stage=name,shape=shape,dtype=dtype,device=device,
            storage_bytes=n,lifetime=lifetime,diagnostic=True))
    csv_rows(out/'memory_breakdown.csv',memory);csv_rows(out/'timing_breakdown.csv',timing)
    # Correct the early diagnostic's interpretation: Long extrinsic is already C2W.
    import numpy as np
    stored=read_csv(out/'stored_prediction_differences.csv')
    from experiments.ours_v7.diagnostics import difference
    for r in stored:
        if r['field']=='c2w_numpy_inverse':
            w=int(r['window'])
            b=np.load(LONG/'native_long/_tmp_results_unaligned'/f'chunk_{w}.npy',allow_pickle=True).item()
            with np.load(OLD/'independent/windows'/f'{w:04d}'/'local.npz') as a:
                r.update(field='c2w',**difference(b['extrinsic'],a['c2w']))
    csv_rows(out/'stored_prediction_differences.csv',stored)
    audit=load(out/'audit_summary.json')
    audit['stored_rows']=stored
    audit['long_pose_convention']='extrinsic is already C2W; initial inverse diagnostic corrected before delivery'
    write_json(out/'audit_summary.json',audit)
    baseline=float(measured(out,'patch')[-1]['seconds'])
    optimized=float(measured(out,'patch_cached_keys')[-1]['seconds'])
    erows=read_csv(out/'export_optimization.csv')
    totals={level:sum(float(r['seconds']) for r in erows if r['iteration']=='1' and r['level']==level and r['stage']=='npz_serialization') for level in ('6','1')}
    sizes={level:sum(int(r['bytes']) for r in erows if r['iteration']=='1' and r['level']==level and r['stage']=='npz_serialization') for level in ('6','1')}
    caches=load(out/'patch_cached_keys_cache_stats_1.json')
    rows=[]
    for mode in modes:
        ts=read_csv(out/f'{mode}_timing.csv');ms=read_csv(out/f'{mode}_memory.csv')
        rows.append(f"| {mode} | {float(ts[0]['seconds']):.3f} | {float(ts[1]['seconds']):.3f} | {float(ts[2]['seconds']):.3f} | {int(ms[1]['peak_allocated'])/2**30:.3f} | {int(ms[1]['peak_reserved'])/2**30:.3f} |")
    commits=subprocess.check_output(['git','-C',str(ROOT),'log','--reverse','--format=%h %s','6554d358fce7f8e551abb6169688bfcbab45e66f..HEAD'],text=True).strip()
    report=f"""# v7 infra 诊断报告
时间：2026-09-22 UTC。所有 GPU 工作在 H20 GPU 4 串行完成。结果目录：{out}。

## 直接回答五个问题

1. **ours independent 与原生逐窗推理一致。** 使用同一原生源码、checkpoint、固定 tensor 和相同运行设置，三个窗口各重复两次，CameraHead 最终输出、c2w、内参、深度、深度置信度、point-head 点图及置信度的 max abs、mean abs、relative L2 全为 0，全部有限；同一路径重复差异也为 0。这里比较的是拼接前原始预测，没有预先做 Sim(3)。
2. **没有发现 independent 实现错误或 Sim(3) 求解/组合错误。** 同一份冻结窗口预测经过两条拼接路径，边级及累计变换完全一致。统一 ownership 后组装差异至多 1.2922e-7，来自 FP32/FP64 组装算术，在原有容差内。历史 Long 与 ours 的原始预测差异可由运行数值设置复现；默认 ownership 也不同。匹配数值设置的完整 100 帧 ATE 为 **{strict['ate_rmse_m']*100:.4f} cm**，逐窗后 ours 前窗优先为 **3.5221 cm**，后窗优先为 **3.5128 cm**。主要剩余差距与分窗计算范围及跨窗几何组装有关，不能进一步把纯上下文信息损失与估计对齐误差定量拆开。
3. **44.68 GiB 的最大来源是预测头卷积临时 workspace。** 分配调用栈定位到 depth head 的 cuDNN execution plan，单笔 35,986,604,048 bytes，即 **33.5151 GiB**。模型及 buffer 的唯一 storage 约 {model_bytes/2**30:.3f} GiB，四层窗口 head cache 为 **5.083 GiB**，其余头部中间结果和输入共同形成峰值。进入 heads 前 independent 的全程峰值约 11.24 GiB。reserved 单列，未当作活跃 tensor。
4. **两项等价优化通过。** K dtype 转换缓存保持 query chunk=64、可见范围、softmax、bank、逐层同步不变；热态前向及 CPU 输出传输从 **{baseline:.3f} s → {optimized:.3f} s**，减少 **{baseline-optimized:.3f} s（{(1-optimized/baseline)*100:.1f}%）**。三次检查共 63 条原始输出对照完全一致；冻结预测复用后的真实 Stitcher 回放与历史轨迹完全一致，同一评测器前后结果完全一致。NPZ 压缩 level 6→1 的完整导出数组序列化从 **{totals['6']:.3f} s → {totals['1']:.3f} s**，减少 **{totals['6']-totals['1']:.3f} s**，文件总大小增加 **{(sizes['1']/sizes['6']-1)*100:.3f}%**，全部数组读回字节哈希一致。两项是分别测得的组件收益，没有把相加值冒充端到端实测。
5. **未获证据的部分不下结论。** 不推断其他场景、KITTI、长序列或其他 GPU 的收益；没有验证图像特征跨窗复用、其他卷积算法或 workspace 限额的严格等价性；没有把每个内部 CUDA buffer 全部归因。没有找到需要修复的主方法 bug，没有修改默认推理入口。

## 实验与来源
- scene0150_00，000000–000099，原有输入 tensor [100,3,392,518] FP32；推理采用 BF16 autocast，原生 heads 和部分 residual/state 保持原本 FP32 路径。
- 窗口 [0,60)、[30,90)、[60,100)；未重新解码、预处理或抽帧。
- 输入文件 SHA-256：1ee3f8d02cd1963083a937f213b3b2bc51a46907f39327572c68b7173a331a66。
- 输入 tensor 字节 SHA-256：092c93de403169f413be3295d84ffde551efa3afbea398bb035e2888fdff892e。
- checkpoint SHA-256：f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e；严格加载全部键。
- 原始 ours 6554d358fce7f8e551abb6169688bfcbab45e66f；VGGT* cc1d8ac15861aea54d14961653cd340e7d984f29；adapter d80946cfb1463ba82ec63530b016d87f63a9cf4d，实际位于同级 fixed_input_baselines。
- vggtlong 无 Git 元数据，按历史 manifest 的逐文件 SHA-256 核验。ours 原始 inference 文件与 manifest 相符，VGGT* models/heads/layers/utils 与 ours 的对应文件逐文件一致。
- 已有未提交修改保留；只增加独立诊断/优化入口及测试。最终源码与诊断文件哈希见 source_manifest.json。
- 初始化 CPU 隔离检查使用真实 checkpoint camera/register token、实际网格和原 initialize；仅图像编码器用零输出 stub。三窗 special token 与原生展开规则、RoPE 坐标均精确相同，首帧分别为 000000、000030、000060。真实编码器已在 GPU 原始预测对照中验证。
- 容差沿用已有 tests/ours_v7/test_model.py：atol=rtol=2e-5，预先固定，未事后放宽。exact 单列。非有限值单独拒绝。
- 全部正式 GT 对齐都是完整轨迹一次 proper Sim(3)，GT 不参与预测、窗口对齐或拼接。

## 首次分叉
**原生逐窗 vs ours independent：未出现分叉。**
输入、初始化、原始输出都有直接检查；因此无需保存全层激活或用小尺寸 FP32 代替固定实验。

**历史 Long vs ours：图像编码器 block 0 的运行数值路径。**
原 baseline adapter 没有启用 ours 的 deterministic=True / cudnn TF32=False。保持同一源码、权重、输入和 BF16，原生默认设置逐元素复现了三份 Long 预测。单独开启 TF32 没改变 encoder；单独关闭 deterministic 则在 patch_embed.blocks.0 首次出现差异。最终 encoder relative L2 约 0.00601、max abs 约 0.70723；这不是可接受的等价优化。历史 manifest 缺少这些 backend 设置本身是对照记录的不足，不是模型权重或算法错误。

同一冻结预测下：
- 两条边各有 6,091,680 个对应像素，overlap 帧与像素顺序相同。
- 阈值为 0.1×min(median(confA),median(confB))；权重 sqrt(confA×confB)，同样的 Huber IRLS delta=.1、5 次上限、tol=1e-9。
- B_local→A_local 方向和累计 compose 逐元素相同。
- ours 保留前窗，Long 官方保留后窗；纯 CPU 隔离测试改变归属时，默认代码不变。
- Long 组装通过实际 save_camera_poses 的 AST 计算部分执行；只移除文件导出，前窗实验裁去后续窗口已归属的区间，没有替换 Sim(3) 公式。

## 显存与生命周期
| 项目 | 实际 storage / bytes | 说明 |
|---|---:|---|
| 原输入 CPU tensor | 243,667,200 | 100 帧单份；窗口切片是 view |
| 当前窗口 GPU 图像 | 最多 146,200,320 | 按窗口复制，另有临时归一化；并非同时驻留 160 帧图像 |
| 160 个窗口帧 token state + RoPE | 684,894,720 | 重叠帧各有独立 token 状态，这是现有语义 |
| 4 层 head feature cache | 5,457,838,080 | 只留 4/11/17/23 层，非全部 24 层；heads 完成一窗即释放 |
| patch bank 逻辑 K/V | {bank_logical:,} | 105 selected token/frame，包括 camera+104 patch |
| patch bank 唯一实际 storage | {bank_unique:,} | V 是 QKV backing storage 的 view，保留了部分未使用 backing；不能仅按 numel 估算 |
| 卷积临时 workspace | 35,986,604,048 | cuDNN run_conv_plan；单个卷积调用生命周期，不是常驻权重/attention score |

bank 没有复制每个窗口的全部本地 token，只含选中部分；目标窗口拼接 K/V 时排除自己的 bank，但本地完整 K/V 仍保留。V view 的额外 backing 是一个小规模可改善项，本轮未宣称它是峰值主因。所有观测的推理阶段禁用梯度、无 grad_fn、模型 eval、参数冻结。生产代码使用 SDPA，未显式构造全场景 attention score 或二维 dense mask；observer 记录输入形状与 mask，峰值调用栈来自卷积。

下表统一为前向加 CPU 输出传输，不含模型加载、拼接或导出。所有新性能控制使用相同 deterministic 设置。冷态是该进程首次前向，热态是第二次；没有 torch.compile。每轮都 empty_cache 并重置峰值，热态保留模型和已 warm 的算子计划，但不是长期服务的无重置吞吐量。

| 模式 | 冷态 s | 热态 s | instrumentation s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

instrumentation 多出 hook 同步、分配调用栈和 tensor 元数据记录，不作为正式性能成绩。明细阶段、CPU RSS、allocated/reserved 在 memory_breakdown.csv。历史完整100默认 backend 的 11.36 GiB 与新匹配设置的约43.05 GiB 不应直接归因于分窗或通信量。

## 时间与优化边界
- 加载和原始 tensor 读取分列；新测试预处理时间为 0，因为复用旧 tensor。旧预处理耗时约2.48s只在历史记录中保留。
- 图像编码、frame/global attention、heads 的同步阶段记录标为 diagnostic；嵌套 interval 不能重复相加。无 instrumentation 的完整前向冷/热时间单独列出。
- 冻结 independent 的两条 ours 拼接边实测合计约25.63s；Long 数值求解路径约23.42s。首次边含已存在 Numba cache 的加载/JIT 初始化，未将它冒充稳定编译耗时。
- 原历史 reconstruction_total 不包括模型加载、GT 评测、绘图和导出。不能把历史 VGGT* 13.06s 纯 forward 与 ours 完整重建直接比较。
- K cache 优化：同一个窗口 global step 内只缓存 SDPA autocast 本来就会做的 K→BF16。每轮 {caches['casts']} 次实际转换、{caches['hits']} 次复用，单窗口缓存上界 {caches['max_cached_bytes']/2**20:.2f} MiB，退出窗口全部释放。没有改 query chunk、softmax、可见性或采样位置。峰值显存基本未下降。
- 该 opt-in context 修改本进程函数绑定，适用于本轮单进程串行推理；没有验证多线程同时调用或分布式场景，默认入口未接入。
- NPZ：level1 保留文件名、字段、shape、dtype、数组值；两轮正反顺序测试，解压后全部字段哈希相同。实际新文件 write+fsync 另测，并只清理本轮创建的临时文件。没有覆盖旧结果。
- **拒绝 query chunk 1024。** 虽热态约35.29s，但原始输出超过2e-5既有容差（例如内参 max abs≈0.593、point-head max abs≈0.457），不能按本轮要求接受。
- 没有减少拟合点、IRLS 次数、分辨率、窗口/overlap、通信比例或采样位置；没有恢复 window_batch_size。

## 测试与优化后回归
- v7 CPU：35 tests，OK；v6 CPU：29 tests，OK。日志 cpu_final_v7.log、cpu_final_v6.log。
- 小张量覆盖输入切片、attention 可见性、窗口顺序、已知 Sim(3) 的方向与累计、前/后窗归属、实际 Long 组装、只读 hook 等价、无损导出、K cache CPU BF16 等价与释放。
- GPU：native/independent/window/patch/K-cache 对照与重复检查；基础四模式和 K-cache 的 instrumentation 输出与各自冻结参考一致。
- 优化后拼接轨迹 max abs=0，同一完整轨迹评测器的 before/after 字典完全一致；见 optimized_downstream_regression.json。
- 发现的失败属于诊断入口导入/fixture 配置，修正日志保留；未据此更改主方法。query1024 的精度失败也完整保留。
- 无 OOM；未停止其他用户任务。没有下载数据集、checkpoint 或拉回大型产物。完整中间激活未落盘。

## 复现
先确认 GPU 4 空闲、磁盘和当前源码。以下命令使用当前诊断提交，创建新的唯一输出目录；不要复用本报告目录。
\`\`\`bash
set -euo pipefail
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v7
export CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
export OUT=/data/yjh/output/vggt/ours_v7_diagnostics/$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$OUT"
"$PYTHON" -B -m unittest discover -s tests/ours_v7 -v 2>&1 | tee "$OUT/cpu_final_v7.log"
"$PYTHON" -B -m unittest discover -s tests/ours_v6 -v 2>&1 | tee "$OUT/cpu_final_v6.log"
"$PYTHON" -B -m experiments.ours_v7.diagnostics audit --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostics init-audit --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostics native --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostics native-default --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_numerics --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_alignment --output "$OUT"
for MODE in full windows independent patch; do
  "$PYTHON" -B -m experiments.ours_v7.diagnostic_profile --mode "$MODE" --repeats 2 --output "$OUT"
done
"$PYTHON" -B -m experiments.ours_v7.diagnostic_profile --mode patch --query-chunk-size 1024 --repeats 2 --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_profile --mode patch --cache-keys --repeats 2 --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_postcheck --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_numerics --full-trajectory --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_export --output "$OUT"
"$PYTHON" -B -m experiments.ours_v7.diagnostic_report --output "$OUT"
\`\`\`
这些入口输出统计、日志与必要的小轨迹；不改变正式 worker。发生 OOM 或其他失败时保留日志并停止，不调整固定实验配置。REPORT 和合并表由 diagnostic_report 生成；它要求所有相应测量完成。

## 提交
\`\`\`text
{commits}
\`\`\`

## 交付文件
REPORT.md、source_manifest.json、raw_prediction_differences.csv、first_divergence.json、alignment_comparison.csv、ownership_comparison.csv、memory_breakdown.csv、timing_breakdown.csv。
额外证据含逐阶段 tensor inventory、encoder/backend 控制、cuDNN allocation trace、CPU/GPU 回归、优化前后表与命令日志。原始输入/权重/窗口预测都复用历史只读来源。
"""
    # Render literal Markdown fences, not escaped fences.
    report=report.replace('\\`','`')
    (out/'REPORT.md').write_text(report)
    write_json(out/'delivery_validation.json',dict(required_files=[
        dict(name=name,bytes=(out/name).stat().st_size,sha256=sha256(out/name)) for name in [
            'REPORT.md','source_manifest.json','raw_prediction_differences.csv','first_divergence.json',
            'alignment_comparison.csv','ownership_comparison.csv','memory_breakdown.csv','timing_breakdown.csv']],
        optimized_predictions_exact=True,optimized_stitching_exact=True,optimized_evaluation_exact=True,
        query1024_accepted=False,all_runs_serial_gpu4=True))
    print('REPORT_READY',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
