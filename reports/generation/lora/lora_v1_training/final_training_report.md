# P6：首版 LoRA v1 正式训练完成报告

状态：`candidate_frozen_pending_project_validation`。完整 3 epochs 正常完成；本阶段结束，不运行项目级效果评测。

## 1. 本轮执行与最终保护

- 用户批准的正式命令：
```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode train --confirm-formal-training
```
- 单次正式进程 PID：`18008`；退出码 0；无中断、无重启、未传入 `--resume`。
- 正式开始前输出目录 `artifacts/lora/lora_v1` 与报告目录 `reports/generation/lora/lora_v1_training` 均不存在；与 P6/P6.2 工程目录隔离。
- 固定本地 base 加载后调用新的 `LoraConfig` / `get_peft_model`；optimizer 与 scheduler 全新构造。初始 global step / micro step 均为 0，没有加载 smoke、low-memory smoke 或 P6.2 状态。
- `latest` 指向最近保存的完整 checkpoint。`best` 仅依据每个 epoch 结束后完整 402 条 `lora_train_dev` loss 更新。唯一 eligible steps：453 / 906 / 1359。
- 本轮无 OOM、非有限 loss/gradient 或 checkpoint 校验故障，没有依据中途 loss 修改任何参数。
- 训练前核对源代码和配置与 P6.2 验收一致；相关单元测试 27/27 通过，`pip check` 通过。本轮未安装/升级依赖。

## 2. 版本与数据冻结

Git HEAD：`08d2e8fdadd3540495563edd2ad3fa5d33e839b4`，分支 `DEV`。LoRA 相关代码尚未提交，本轮未 commit / push；不能只用 HEAD 代表实际训练代码。

独立 [pre-training manifest](../lora_v1_pretraining_manifest.json) 保存训练代码、配置等文件的原始字节 Base64 快照和 SHA-256，同时记录 Git 工作区、环境、Prompt/chat-template 与固定基座所有 11 个文件的哈希。SHA-256：

```text
4b4e70f00005864b4310a7a80a285c833ec13d24b983e01447dcabca8661778a
```

正式配置仍为 P6.2 验收版本：`configs/lora_qlora_train_v1.json`，SHA-256：

```text
1afb8ed3ecca91aa3d8bdd55fb5e456dbb9a60deef8d52f4805c70cc24d7d1e0
```

配置中的旧 `pending_final_full_training_confirmation` 状态文字没有修改，以保持验收 hash；本轮正式授权已记入 pre-training / final manifest。

| 数据职责 | 实际路径 | 商品数 | 任务数 | SHA-256 |
| --- | --- | ---: | ---: | --- |
| lora_train：参数更新 | data/processed/lora_instruction_data_v1/train.jsonl | 1206 | 3618 | c31b00f3d19d503b3f7c7529cc2ce7e115143214b0d9ab40f07412becd6a5434 |
| lora_train_dev：监控与选择 | data/processed/lora_instruction_data_v1/validation.jsonl | 134 | 402 | 20a32d5f67a8ac6dd0743bb90445522fe28cd4099ed318e71a4f0e8e33fa0a94 |

注意：上述 `validation.jsonl` 是 P5 内部的 `lora_train_dev`，不是 `project_validation`。原冻结项目级 8:1:1 划分未改变，没有新增 test split。

P5 数据 manifest：`reports/generation/lora/lora_instruction_data_v1_final_manifest.json`，SHA-256 `e21b9f278dbe533f038008840bce313564e702b6b5ff7ba5e0f8714156814a5d`。
P5 模板版本 `lora_copy_templates_v1`；只使用冻结 messages，原标题/target 不进入生成输入。
Qwen chat-template SHA-256 `2e2d2512cfe46af53dc1eed45368ecaab26ac4461c480e6b69fe571cf0ceaa75`，使用训练入口相同的 JSON 序列化算法核验。

训练结束后再次核对：P5 train/dev、pre-training 快照文件和所有基座文件哈希均未变化。

## 3. 实际训练配置（全程未改变）

| 参数 | 本轮值 |
| --- | --- |
| base model | Qwen/Qwen2.5-7B-Instruct |
| revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| 加载 | 固定本地 snapshot；HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、local_files_only=True |
| 量化/精度 | NF4 4-bit；double quant；BF16 compute/autocast；冻结 embedding/head BF16；norm/LoRA FP32 |
| LoRA | q_proj/v_proj；rank=8；alpha=32；dropout=0.05；bias=none |
| 可训练参数 | 2,523,136，仅 LoRA |
| loss | 冻结的 native-shape BF16 logits + assistant-only causal FP32 CE，chunk=16 |
| micro batch / accumulation | 1 / 8；每 epoch 末尾 2 条按实际条数归一化 |
| 序列 | max=320；无截断；动态长度、无固定 padding |
| optimizer | torch.optim.AdamW；lr=2e-4；betas=(0.9,0.999)；eps=1e-8；weight_decay=0.01 |
| scheduler | constant_lambda_1；warmup=0；未启用梯度裁剪 |
| epochs / early stopping | 3 / 不启用 |
| seed / 顺序 | 42；逐 epoch 本地 random seed+epoch shuffle |
| gradient checkpointing | true；use_reentrant=false；use_cache=false |
| dev | 每 epoch 结束完整 402 条无梯度 evaluation；按最低 dev loss 选 best |
| checkpoint | 每 151 optimizer steps 及 epoch-end；只保留 latest/best；不保存完整 7B |

实际所有日志学习率均为 `0.0002`，scheduler step 与 global optimizer step 一致。
每 epoch：3618 micro steps / 453 optimizer steps；合计 10854 / 1359。

## 4. 每 epoch 结果

| Epoch | 平均 train loss | 完整 dev loss | dev 条数 | 纯训练耗时 | dev 耗时 | epoch 墙钟耗时 | train 峰值分配显存 | dev 峰值分配显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.052169411 | 0.019772963 | 402/402 | 31.48 min | 72.52 s | 32.73 min | 5661.9 MiB | 5535.6 MiB |
| 2 | 0.011435231 | 0.017524495 | 402/402 | 31.64 min | 73.12 s | 32.90 min | 5661.9 MiB | 5535.6 MiB |
| 3 | 0.008005011 | 0.013250582 | 402/402 | 31.95 min | 74.15 s | 33.23 min | 5661.9 MiB | 5535.6 MiB |

训练总墙钟耗时：**5947.31 秒 / 99.12 分钟**，包含初始化、dev、保存与日志等。
epoch 墙钟耗时从训练 start/前一次 epoch-end 到本次 epoch-end；checkpoint 保存、日志等开销另见时间戳，不能将纯训练耗时直接当作总耗时。
显存均为 PyTorch allocator 指标，不是整个 Windows/GPU 的总占用。train/dev 峰值 reserved 均为 5774.0 MiB；训练峰值 allocated 5661.9 MiB（约 5.53 GiB），dev 为 5535.6 MiB（约 5.41 GiB）。

train loss：该 epoch 全部 3618 条 assistant-token mean causal CE 的样本平均。
dev loss：完整 402 条相同口径的样本平均；三个 task 各 134 条。每次均 no_grad、parameter_updates=0、模型工程配置未改变。

| Epoch | dev title loss | dev selling_points loss | dev short_description loss |
| --- | ---: | ---: | ---: |
| 1 | 0.056509932 | 0.001656897 | 0.001152061 |
| 2 | 0.044270904 | 0.000630812 | 0.007671770 |
| 3 | 0.039231254 | 0.000103205 | 0.000417287 |

这些数值仅说明对 P5 target 的拟合，不等同于核心属性命中率、通顺率或事实错误率，也不能代替 Base vs LoRA 项目级效果评测。

## 5. Best / latest 与独立加载验收

best：**epoch 3 / step 1359 / dev loss 0.013250582**，是三个完整 epoch-end evaluation 的最小值。

best 与 latest 相同：
```text
artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359
artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter
artifacts/lora/lora_v1/best.json
artifacts/lora/lora_v1/latest.json
```

adapter_model.safetensors：10,107,280 bytes。SHA-256：
```text
890a061812321ccf6bbf598b37617fc59dcde70d57042c4304c219a0f6740e31
```

checkpoint 包含 adapter、optimizer、scheduler、epoch/global step/cursor、Python/Torch CPU/CUDA RNG、配置快照和完整校验 manifest。

按 latest/best 策略，本轮 8 个已被替代的 checkpoint（151、302、453、604、755、906、1057、1208）已清除，不经过回收站、未保留副本。最终 1359 完整保留，P6/P6.2 工程 checkpoint 未触碰。

独立验证命令（本轮已执行一次）：
```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -X utf8 scripts/verify_lora_v1_best.py
```

结果见 [best_adapter_reload.json](best_adapter_reload.json)：

- 新进程 PID `29200`，与训练进程不同；耗时 40.58 秒。
- 重新从固定 snapshot 加载 4-bit base，再加载 best adapter；不加载 optimizer/scheduler，不更新参数。
- 所有保存/加载后的 adapter 张量逐项完全一致。
- 冻结 embedding/head BF16、norm FP32、use_cache=false，与冻结工程路径一致。
- 仅选 P5 `lora_train` 第一个商品 `14622014051` 的 title / selling_points / short_description 三条任务；有限 forward loss，三个非空推理输出。
- 推理 greedy、max_new_tokens=128，仅机械验收；不做效果判断、不当作训练真值、不属于项目级评测。
- 本验证脚本是单独的收尾工具，其 SHA 已进入最终 manifest；没有修改 pre-training 冻结的训练入口与 loss 代码。

## 6. 最终归档与建议 Git 范围

- [final_manifest.json](final_manifest.json)：best adapter、checkpoint、配置、代码、P5 数据、环境、基座、Prompt、报告/日志的版本与 hash 关系。
- [final_checksums.sha256](final_checksums.sha256)：包括本报告与 final manifest 在内的文件 SHA-256；manifest 不自包含 hash，避免循环。
- 正式训练日志：`events_20261004T064046462933.jsonl`。
- 完整运行记录：`invocation_20261004T064046462933.json`。
- 完整 dev 记录：`dev_epoch_1.json`、`dev_epoch_2.json`、`dev_epoch_3.json`。
- 环境 freeze：`environment_20261004T064046462933.txt`。

建议后续由用户确认后提交：P5 构造代码/配置/测试及小型报告；P6 省显存 helper/smoke 配置/脚本/测试及小型报告；P6.2 正式入口/配置/测试/验收报告；本轮独立验证脚本和 pre-training/final manifest、报告、loss 日志、环境 freeze、校验清单。
无需自动纳入与本轮无关的 Colab 文档；大数据、基座缓存、adapter 权重、optimizer/RNG 二进制状态不提交 Git，应另行备份整个最终 checkpoint。
本轮未 commit / push，未修改依赖清单或 P5 文件。

## 7. 停止边界

- project_validation：本轮未读取/未运行。
- project_test：本轮未读取/未运行。
- RAG v2 32 条 generation-output holdout：保持 frozen_not_executed；本轮连其 manifest/checksum 内容也未打开。
- smoke/P6.2 结果保留，不重写既有结果。
- 当前候选按内部 dev loss 冻结，尚无项目级效果结论。
- **到此停止，等待用户确认训练结果；确认后再准备/执行 Base vs LoRA 同配置 project_validation 评测。本轮不继续。**

