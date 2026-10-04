# P6 本地 7B 省显存 smoke 与最长样本检查

## 结论与停止点

本机 RTX 4070 Laptop 8GB 上，独立省显存版本通过工程检查。无需削减 P5 数据、降低 rank 或换成小模型。当前只验证工程可运行性，不判断 LoRA 效果，不启动正式训练。

最终运行以 `p6_low_memory_smoke_v1_retry4/smoke_report.json` 为准。正式工程参数已整理到 `configs/lora_qlora_local_training_candidate_v1.json`，该文件明确不可执行训练；学习率、epoch、scheduler、保存/评测频率和早停细节仍待确认。仓库仍没有正式 trainer / resume / dev evaluation 入口，本次未扩展实现它们。

## 范围和隔离

- P5 文件、数据版本及划分不变。`lora_train` 仍为 1,206 件 / 3,618 条任务，`lora_train_dev` 为 134 件 / 402 条任务。
- `validation.jsonl` 是已有内部 `lora_train_dev` 文件名，不是 `project_validation`，不改名或重新划分。
- 参数更新只在冻结 `lora_train` 的固定 24 条 smoke 和固定最长 8 条压力样本上进行；压力样本重复 8 次。不遍历完整 epoch。
- 为选取最长样本，仅对全部 3,618 条 `lora_train` 做 CPU tokenizer 审计，不截断、不改写。最长 249 tokens，现有 max length=320 足够。
- 本次不执行 dev 评估，仅校验其文件哈希；后续训练监控、checkpoint 选择、early stopping 使用 `lora_train_dev`。
- `project_validation` 仅在训练方案和候选 adapter 冻结后进行项目级效果验证，不用于日常训练调参。
- 未读取 `project_test` 样本、未执行 RAG generation-output holdout、未创建 formal test100。holdout 清单仅作哈希完整性检查。
- 未安装、升级依赖，未修改 requirements，未 commit / push。旧 P6 配置、入口及结果保持原样。

## 实现和精度

新增入口：`scripts/run_qlora_low_memory_smoke.py`；新增模块：`src/generation/qlora_memory.py`；新增 smoke 配置：`configs/lora_qlora_low_memory_smoke_v1.json`。复用旧入口的选择/tokenizer/hash 等工具，不修改旧入口。

1. 固定本地 Qwen2.5-7B-Instruct snapshot，revision=`a09a35458c702b33eeacc393d103063234e8bc28`。进程内启用离线模式，直接读取 snapshot，禁止隐式下载。
2. NF4 + double quant + BF16 compute。PEFT 准备阶段把冻结 embedding/head 暂存 CPU，让临时 FP32 转换不占 GPU；恢复 BF16 后搬回原设备。仅改变初始化存储位置。
3. 冻结 embedding/head 保持 BF16；norm 和可训练 LoRA 保持 FP32；前向使用 BF16 autocast。相对 PEFT 的两个 FP32 大层，存储减少 2,179,989,504 bytes，约 2.03 GiB。
4. 保留原输出 head 的 BF16 投影形状。仅对受监督的 assistant token 计算分块 FP32 CE，chunk=16；checkpoint 只保留低精度 scores，不保留完整序列 FP32 logits/CE 图。
5. 因果 shift、prompt=-100、有效监督 token 分母与原损失相同；每样本 mean loss，再按累积=8 缩放。不是修改属性命中率等正式生成评测口径。
6. decoder checkpoint 使用 `use_reentrant=false`；`use_cache=false`。梯度等价检查开启 decoder checkpoint，仅暂时关闭 dropout，检查后训练阶段恢复原 dropout=0.05。

精度变化本身不宣称 bitwise 等价：初始固定样本 loss 从 0.8745776415 变为 0.8883368969，绝对差 0.0137592554，小于预先设置的工程阈值 0.05。此项是数值检查，不是效果合格证。CPU 暂存初始化与已验证 BF16 初始 loss 一致。

在相同 BF16 精度下，分块 loss 与标准 loss 均为 0.8883368969，差异为 0；实际所有 LoRA 参数的梯度最大绝对差及相对 L2 差异均为 0。这是单个固定实际样本的 GPU 对照；CPU 单元测试另覆盖多种 chunk、batch 和内部忽略标签，不等同于验证全部训练样本逐项 bitwise 一致。

## 最终实测

| 项目 | 结果 |
| --- | --- |
| LoRA 配置 | q_proj / v_proj，rank=8，alpha=32，dropout=0.05 |
| 可训练参数 | 2,523,136，与旧 smoke 相同 |
| 固定 smoke | 原来的同一 24 条任务，24 micro / 3 optimizer steps |
| 最长样本压力 | 全部训练任务最长前 8 条，长度 241–249，重复 8 次；64 micro / 8 optimizer steps |
| loss / 梯度 / 参数更新 | 全部 finite，optimizer 确实更新 adapter |
| 旧 smoke 峰值 allocated | 8,423.5 MiB；旧报告口径 |
| 新 smoke 峰值 allocated / reserved | 5,623.8 / 5,754.0 MiB |
| 最长样本峰值 allocated / reserved | 5,661.3 / 5,884.0 MiB，allocated 约 5.53 GiB |
| 初始化峰值 allocated | 5,425.5 MiB |
| loss/梯度对照峰值 allocated | 5,730.3 MiB；独立诊断阶段，不并入训练峰值 |
| 重载峰值 allocated | 5,522.3 MiB |
| 累积边界 allocated 增长 | 0.0 MiB |
| NVIDIA 全阶段最低采样 free | 1,275 MiB，约 1.25 GiB |
| Windows 全阶段进程共享显存最大采样值 | 78 MiB |
| smoke / 压力耗时 | 12.67 / 38.63 秒 |
| 压力 micro step 中位耗时 | 0.598 秒 |
| adapter 保存及重载 | 权重逐项一致，重载前后 loss 均为 0.2059225440 |
| 测试 | `python -m pytest -q`：144 passed |

新旧 smoke 峰值 allocated 减少约 2.73 GiB（33%）。不同阶段的峰值分别统计，不能把初始化/等价检查峰值冒充训练峰值，也不能把 allocated 当成驱动物理显存占用。

工程护栏为：采样 free>=512 MiB、进程 shared<=128 MiB、累积边界 allocated 增长<=128 MiB。它们是本次新增的保守工程阈值，不是 PRD 效果指标。最终训练阶段和包含初始化/重载的全阶段均通过；没有放宽阈值。

计数器不是瞬时峰值证明，NVIDIA 和 Windows 计数器也不是同步采样。78 MiB 共享占用不等同于模型权重溢出证明，不能宣称共享显存绝对为零。64 连续 micro steps 是约 39 秒的短时稳定性检查，不是整 epoch、长时间温度/功耗或断点恢复保证。

## 保留的工程尝试

| 目录 | 结论及原因 |
| --- | --- |
| `p6_low_memory_smoke_v1` | 首次尝试未更新参数：投影按监督 token 分块，loss 相同但 LoRA 梯度相对 L2 差异 1.036%；严格检查拒绝 |
| `p6_low_memory_smoke_v1_retry1` | 保留原 BF16 head 投影后梯度差异为0；24+64步、重载通过；初始化 FP32 临时分配/共享显存未满足保守护栏 |
| `p6_low_memory_smoke_v1_retry2` | CPU 暂存避免 FP32 大层 GPU 临时副本；训练通过；等价检查整体 eval 关闭 decoder checkpoint，诊断峰值仍偏高 |
| `p6_low_memory_smoke_v1_retry3` | 等价检查保留 checkpoint；全部护栏通过；发现循环局部变量保留旧 head，使重载多保留约1GiB |
| `p6_low_memory_smoke_v1_retry4` | 释放旧模块引用；重新验证同一范围，作为最终工程结果 |

这些是实现/数值/资源检查，不是数据或内容 Prompt 调参。所有已落盘结果不覆盖。retry1–3 的实际代码/config 快照保存在对应 `source_snapshot/` 和 `config_snapshot.json`；快照用于哈希审计，不能直接从 reports 目录执行它们。

## 正式候选参数与未决项

工程建议沿用：固定模型 revision；NF4/double quant；BF16 compute/autocast/冻结大层；FP32 norm/LoRA；q_proj/v_proj；rank=8、alpha=32、dropout=0.05；micro batch=1、累积=8；动态实际长度、不补齐到320；max length=320、禁止截断；seed=42；非重入 checkpoint；use_cache=false；chunk CE=16；AdamW。

不需要为了省显存再裁减 P5 商品数，也不需要新增8-bit optimizer依赖。max length 是上限，降低它不会缩短当前已经按实际长度处理的样本；截断会损伤监督事实，不能用作不透明的省显存手段。

学习率 2e-4 只沿用为 smoke 参考，不是已经确认的正式值；epoch/max steps、scheduler/warmup、weight decay、grad clipping、最后不足8条的累积分母、save/eval frequency、checkpoint retention、early stopping patience/min delta、恢复策略均需确认。既有 Colab 指南中的 save/eval=50、保留2个 checkpoint 也只是建议，尚未获批准。

建议监控 `lora_train_dev` 的逐样本 assistant-token mean CE，再对样本求平均，另报告三个 task_type 的 loss。checkpoint 选择与早停只用这一内部开发集；loss 不替代项目级通顺率、属性命中和事实错误评测。保存 adapter + optimizer/scheduler/RNG/state，不重复存整个7B。正式初始化必须来自固定基座，不得使用本次 smoke adapter。

## 执行入口、归档与 Git

本次实际执行命令（PowerShell、项目根目录）：

```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/run_qlora_low_memory_smoke.py --config configs/lora_qlora_low_memory_smoke_v1.json
python -m pytest -q
```

smoke 拒绝覆盖已有输出。最终配置目前指向已存在的 retry4，不能原样再次执行；若将来需要重跑，先另行确认新的输出位置，不能删除旧结果凑重跑。

版本/hash 关系在 `p6_low_memory_smoke_v1_archive_manifest.json` 和 `p6_low_memory_smoke_v1_checksums.sha256`。其中包含 P5 manifest 指向的文件核验、旧 P6 冻结边界、最终代码/config/report/telemetry/environment/tests/adapter 的 SHA-256，以及保留的尝试。

建议本次 Git 范围仅包含：两个新增 configs、新 smoke 入口、新模块、新测试、本阶段小型 JSON/Markdown/freeze/checksums 与必要快照。`artifacts/lora/` 内的 smoke adapter 不提交 Git，也不用于正式训练；大模型缓存和 P5 数据仍保留已有忽略规则。原来未提交的 P5、旧 P6 和 Colab 指南不由本次自动提交。

当前停止。下一步先确认正式训练未决项，再单独实现具有 dev evaluation、保存/恢复与完整 manifest 的最小正式训练入口；本阶段不替用户启动训练。
