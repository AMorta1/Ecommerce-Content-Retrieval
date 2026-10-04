# P6.2 首版 LoRA 正式入口准备与验收

## 结论与边界

P6.2 工程验收通过。独立正式训练入口、checkpoint/恢复、完整内部 dev evaluation 已实现并验证；**完整3-epoch训练没有启动**，等待用户最终确认。

新入口 `scripts/train_lora.py` 调用 `src/generation/lora_training.py`。没有 import、调用或改写 smoke trainer；只复用冻结的 `src/generation/qlora_memory.py` 中省显存初始化和 loss。P5、旧 smoke、旧候选配置和报告均未修改。本次新增 `configs/lora_qlora_train_v1.json` 才是正式入口使用的完整候选配置。

执行模式为：默认 `prepare` 只验证配置/数据 hash 并计算计划；`acceptance` 的 global step 上限硬限制为4，独立输出目录；`train` 必须显式传 `--confirm-formal-training`。本轮实际运行全部是 acceptance，没有使用该正式训练确认参数。

## 数据隔离与数值口径

| 名称 | 实际文件 | 商品 / 任务样本 | 职责 |
| --- | --- | --- | --- |
| `lora_train` | `data/processed/lora_instruction_data_v1/train.jsonl` | 1,206 / 3,618 | 唯一参数更新来源 |
| `lora_train_dev` | `data/processed/lora_instruction_data_v1/validation.jsonl` | 134 / 402 | 无梯度 dev loss；正式训练时选择 checkpoint |
| `project_validation` | 本阶段不打开 | 原项目级冻结集合 | 首版候选冻结后才验证项目效果 |
| `project_test` / RAG holdout | 本阶段完全不打开，包括不做其文件 hash 读取 | 原冻结集合 | 不进入训练或本轮验收 |

`validation.jsonl` 是 P5 原文件名，含义仍为内部 `lora_train_dev`，不代表项目级 validation。没有重抽、重切分或改写 P5。train/dev 路径及 hash 必须匹配白名单，否则在读取样本前报错。

更新路径只索引 `train_items`。dev 样本采用 lazy loading，只在无梯度 evaluation 时解析/使用；启动和结束的 dev 字节 hash 核验仅用于完整性，不参与梯度。evaluation 检查全部参数的版本计数不变、梯度均为 None、精度/量化/LoRA配置签名不变，并恢复 RNG 与原 train/eval 模式。

输入/target直接使用冻结 messages，原 source_title 不重新构造输入；tokenizer/chat template来自固定 snapshot，prefix、assistant-only mask、因果 shift和有效 token 分母与旧实现相同。动态实际长度，不padding到320，不截断。每条任务 loss 是受监督 token 的 mean CE，训练累积按样本均值；dev 对402个逐样本 mean CE再平均，另报三任务均值。这不是属性命中率/事实错误等生成评测，也不作为模型效果结论。

## 完整候选配置

完整 JSON：`configs/lora_qlora_train_v1.json`，SHA-256=`1afb8ed3ecca91aa3d8bdd55fb5e456dbb9a60deef8d52f4805c70cc24d7d1e0`。

| 参数 | 候选值 | 来源 |
| --- | --- | --- |
| Base | Qwen/Qwen2.5-7B-Instruct | 已冻结工程参数 |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` | 已冻结工程参数 |
| Snapshot | `E:\Projects\Baidu\.model-cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28` | 当前真实本地缓存 |
| 网络 | process内 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE=1，直接snapshot，local_files_only=true | 本轮明确要求 |
| Prompt/template | P5 `lora_copy_templates_v1`，原messages；固定tokenizer_config的chat template | 保持不变 |
| Quantization | NF4 4-bit，double quant=true | 已验证，不变 |
| Precision | BF16 compute/autocast/冻结embedding/head，FP32 norm/可训练LoRA | 已验证，不变 |
| 初始化 | PEFT准备时CPU暂存两个冻结大层，再BF16回原GPU | 已验证，不变 |
| LoRA | q_proj/v_proj，rank8，alpha32，dropout0.05，bias none | 已验证，不变 |
| 可训练参数 | 2,523,136 | 与原smoke一致 |
| Batch | micro1，accumulation8，常规effective batch8 | 已验证，不变 |
| 尾组 | 最后2条按实际2条归一化，不丢弃、不补重复、不跨epoch混合 | 新trainer必要局部实现 |
| Max sequence | 320，dynamic length，禁止截断/固定padding | 已验证，不变 |
| Loss | 原native-shape BF16 projection + assistant-only chunked FP32 mean CE，chunk16 | 模块原hash保持不变 |
| Gradient checkpointing | true，use_reentrant=false，use_cache=false | 已验证，不变 |
| Seed | 42 | 已验证，不变 |
| Epoch shuffle | 本地独立Random(seed+epoch)，每条train任务恰好一次 | 新trainer确定性样本顺序 |
| Optimizer | torch.optim.AdamW；betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01 | 显式记录原smoke默认值，不另调参 |
| Gradient clipping | 不启用 | 沿用smoke |
| Learning rate | 2e-4 | 用户本轮候选 |
| Epoch | 3 | 用户本轮候选 |
| Scheduler | 恒定LambdaLR(lambda=1)，warmup0 | 新trainer建议；不改变原恒定LR，支持state保存/恢复 |
| Early stopping | 不启用 | 用户本轮候选 |
| Dev evaluation | 每epoch结束一次，完整402条，batch1/no_grad | 用户本轮候选 |
| Selection | 最低整体dev loss；相等保留较早checkpoint | 用户本轮候选 |
| Save | 每151 optimizer steps，且epoch末保存 | 本轮算出的建议 |
| Retention | 正式运行只保留完整latest和best；acceptance保留其有限checkpoint | 用户候选与验收可追溯性 |
| 输出 | `artifacts/lora/lora_v1/` | 新正式运行目录，当前未创建 |
| 报告 | `reports/generation/lora/lora_v1_training/` | 新正式运行目录，当前未创建 |

这些是待最终确认的正式候选，不是已启动的实验。Scheduler、尾组归一化和明确化的optimizer默认值不是新一轮显存调参。

## Steps、checkpoint间隔和耗时

- 每epoch：3,618 micro steps；`ceil(3618/8)=453` optimizer steps，其中452组8条、最后1组2条。
- 3 epochs：10,854 micro steps / 1,359 optimizer steps。
- Save steps=151：epoch1在151/302/453保存，epoch2在604/755/906，epoch3在1057/1208/1359；恰好每epoch3次。
- Dev：453/906/1359 step时，各完成一次完整dev，再把当前checkpoint与历史best比较。
- 本次每optimizer组约4.27秒，因此151步约10.7分钟，提供合适恢复粒度；无须高频保存小于1分钟的checkpoint。
- 实测连续32个train micro steps耗时17.0849秒，平均0.5339秒/条；纯3-epoch更新粗估96.6分钟。
- 先前最长样本压力中位0.5983秒/条作为保守参考，纯更新约108.2分钟。
- 完整dev耗时72.26秒，3次约3.61分钟；合计约100.2–111.8分钟，另加加载、日志、checkpoint I/O和笔记本温度/功耗波动。建议至少预留2小时，必要时留2.5小时。不是已完成完整epoch的实测。

## 断点恢复验收

真实三个独立进程：连续4步 PID30668；先2步 PID6932；resume续2步 PID20612。没有在同一进程中用内存对象冒充重启。

| 项目 | 结果 |
| --- | --- |
| 恢复起点 | epoch0，global step2，cursor16，scheduler last_epoch2 |
| 续训终点 | epoch0，global step4，cursor32，global micro32，scheduler last_epoch4 |
| Shuffle顺序 | 逐条商品/task与连续运行一致 |
| Loss链路 | 32条整体逐样本loss最大绝对差0；其中恢复后的16条差0 |
| 最终Adapter | 两路safetensors SHA完全一致：`4360f91f7250351ace2de0af732b3da2f1a57bda0b630d3a77c40c484e4adb30` |
| Optimizer / scheduler / RNG | 最终内容digest均一致；加载瞬间state也经过严格核验 |
| Train峰值allocated | 5,647.7 MiB，约5.52 GiB |
| LR | 一直2e-4 |

工程checkpoint分别在 `artifacts/lora/p6_2_acceptance_v1/uninterrupted/` 和 `.../split/`，正式运行禁止从 acceptance或smoke adapter初始化。mode/identity不匹配直接拒绝。此验收不构成LoRA效果实验，也不挑选这几个工程adapter作为候选。

Checkpoint包含adapter、optimizer/scheduler state、epoch/cursor/global steps、epoch累计loss/计数、累计训练耗时、CPU/CUDA/Python RNG，以及配置/data/template/model/env/code身份。只在optimizer边界保存，不需要恢复未提交的累积梯度。全部文件带SHA，完整目录原子发布后更新latest指针；`.incomplete`不允许恢复。adapter保存不包含整7B，optimizer等完整续训状态另存。PEFT的adapter格式可见[官方说明](https://huggingface.co/docs/peft/developer_guides/checkpoint)。

恢复必须来自同一个run的完整latest；先验证manifest及全部文件，要求config、源码、环境版本、base snapshot、chat template一致，再恢复adapter/optimizer/scheduler/RNG并按cursor重建同一epoch顺序。不能一边更改正式配置一边声称确定性resume。本次证明的是当前硬件/软件环境的短序列一致性，不保证跨版本/跨硬件bitwise一致。

正式retention会删除这个run/checkpoints目录内过期、完整、非latest/best的checkpoint，且先验证解析后路径位于指定目录；不碰用户目录或不符合命名的内容。本轮仅在临时测试夹具验证清理，没有删除用户模型结果。acceptance的2/4步checkpoint保留可审计。

## 完整内部dev验收

同一训练模型在resume第4步之后无梯度评估全部402条；未为dev改变NF4、rank、精度、loss、batch或max length。

| 项目 | 实测 |
| --- | --- |
| 样本数 | 402 / 402，134商品×3任务 |
| 最长序列 | 269 tokens，320上限内，未截断 |
| dev loss | 0.9523777202 |
| title loss | 1.1990867375 |
| selling_points loss | 0.9366371761 |
| short_description loss | 0.7214092470 |
| 耗时 | 72.2633秒 |
| 峰值allocated / reserved | 5,535.6 / 5,772.0 MiB，allocated约5.41 GiB |
| 参数更新 / 梯度 | 0 / None；no_grad=true |
| 配置 / RNG / model mode | 检查不变/恢复完成 |

此dev loss只是工程通路结果，不是首版完整LoRA的最终dev指标，不用于比较生成质量或证明品类表达效果。正式训练时仅每epoch末dev loss决定best，不开启early stopping。

## 你确认后才运行的正式命令

### 开始正式3-epoch训练

在PowerShell输入以下命令；**本轮不要执行**：

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode train --confirm-formal-training
```

### 断线/中断后恢复

重新打开PowerShell，读取真实latest指针，不能猜checkpoint编号：

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
$loraResumePath = (Get-Content 'artifacts/lora/lora_v1/latest.json' -Raw -Encoding UTF8 | ConvertFrom-Json).path
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode train --confirm-formal-training --resume $loraResumePath
```

不要用工程checkpoint作正式初始化。不要把best拿来覆盖latest续训；best用于最终候选选择，latest用于续训。不要原样fresh重跑已有输出目录：入口拒绝覆盖。若checkpoint保存中断产生`.incomplete`或校验失败，先汇报，不自动删除/覆盖修复。

正式完成后读取 `artifacts/lora/lora_v1/best.json` 的真实path，候选权重是该checkpoint里的 `adapter/adapter_model.safetensors` 和 `adapter_config.json`；保留对应配置、状态、环境和报告。先冻结首版，再得到单独确认运行 `project_validation`，与同配置未微调base对照；本阶段不运行 `project_test`。

## 本轮实际执行的验收命令

以下已执行，不是完整训练：

```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode acceptance --stop-at-step 4 --output-directory artifacts/lora/p6_2_acceptance_v1/uninterrupted --report-directory reports/generation/lora/p6_2_acceptance_v1/uninterrupted
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode acceptance --stop-at-step 2 --output-directory artifacts/lora/p6_2_acceptance_v1/split --report-directory reports/generation/lora/p6_2_acceptance_v1/split
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode acceptance --stop-at-step 4 --resume artifacts/lora/p6_2_acceptance_v1/split/checkpoints/checkpoint-step-000002 --output-directory artifacts/lora/p6_2_acceptance_v1/split --report-directory reports/generation/lora/p6_2_acceptance_v1/split --evaluate-dev
python -m pytest -q
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -m pip check
```

20个新增测试，全套164 passed；pip check通过。验收更新总共8个optimizer steps（连续对照4 + 分段2+2），仅触达32条不同train任务；没有执行完整epoch。402条dev仅在最后进程evaluation一次。本轮未运行任何文案生成，只计算训练/评估loss。

## 归档和停止

`p6_2_acceptance_v1/acceptance_summary.json` 是统一验收数据；三个invocation报告保留独立PID、step日志、模型/config/env身份。`p6_2_trainer_archive_manifest.json` / `p6_2_trainer_checksums.sha256` 串联新增源码、配置、测试、报告、checkpoint和P5冻结边界；其中不需要打开测试/holdout文件。

建议提交本轮新增配置、独立入口、module、tests及小型reports/manifests/environment记录。`artifacts/lora/p6_2_acceptance_v1/` 的工程权重和optimizer状态不提交Git，不复用为正式训练初始化。旧P5、旧P6和Colab指南的既有未提交内容不由本轮自动提交。没有安装依赖、commit或push。

**P6.2验收完成后停止。等待用户最终确认完整候选配置及3-epoch启动，不自行训练或扩大评测范围。**
