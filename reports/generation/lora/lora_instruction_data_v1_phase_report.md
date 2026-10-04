# P5 LoRA v1 指令数据构造报告

## 结论

P5 数据构造与样例审核已完成，可以作为后续 QLoRA 工程 smoke 的候选数据版本；本阶段未安装依赖、未加载训练模型、未启动训练，也未读取或运行冻结 test。

LoRA v1 的职责是学习三类电商内容的表达结构和品类话术。事实约束仍由可靠输入属性和后续 RAG/生成链路负责，不把“降低事实幻觉”作为本 LoRA 数据版本的主要训练目标。

## 数据构造方案

数据源仅为 `week2_train_v1` 的1,559件训练商品。处理顺序如下：

1. 排除15件已确认 CONFLICT，暂缓28件 REVIEW；
2. 使用冻结 generation evaluation 中的品类核心属性定义；
3. 对占位值、非信息型型号、多值 SKU、字段类型、品类标题和配件边界做确定性门控；
4. 要求每件商品至少有3个可主动表达的可靠核心属性；
5. 原标题只用于安全组件排序、风格参考和质量审查，不进入模型输入；
6. title 从品类和可靠属性组件中按原生标题顺序压缩重组；
7. selling points 固定为3个中性字段值，不添加效果、评价或场景；
8. short description 用少量中性连接词串联最多5个可靠属性；
9. 每件合格商品拆成 `title`、`selling_points`、`short_description` 三条独立指令；
10. 所有任务的模型输入统一为品类、可靠核心属性和 task type。

没有使用 Baseline、RAG v1、RAG v2 或任何其他模型生成结果作为训练真值。

## 最终规模

| 项目 | 数量 |
|---|---:|
| 源训练商品 | 1,559 |
| 最终可用商品 | 1,340 |
| 训练商品 | 1,206 |
| LoRA validation 商品 | 134 |
| 总指令样本 | 4,020 |
| title | 1,340 |
| selling_points | 1,340 |
| short_description | 1,340 |

### 各品类分布

| 二级品类 | 可用商品 | Train 商品 | Validation 商品 | 三任务样本总数 |
|---|---:|---:|---:|---:|
| 保温杯 | 184 | 166 | 18 | 552 |
| 垃圾桶 | 176 | 158 | 18 | 528 |
| 拖把 | 156 | 140 | 16 | 468 |
| 收纳箱 | 149 | 134 | 15 | 447 |
| 移动电源 | 172 | 155 | 17 | 516 |
| 耳机 | 171 | 154 | 17 | 513 |
| 键盘 | 154 | 139 | 15 | 462 |
| 鼠标 | 178 | 160 | 18 | 534 |

### 过滤与暂缓原因

| 原因 | 商品数 | 处理 |
|---|---:|---|
| 已确认 CONFLICT | 15 | 排除 |
| REVIEW 待外部证据 | 28 | 暂缓 |
| 可靠核心属性少于3项 | 50 | 排除，不补写事实 |
| 品类标题缺少对应商品语义 | 56 | 排除 |
| 配件或商品边界风险 | 39 | 暂缓 |
| 最终事实签名重复 | 31 | 确定性去重 |
| 合计 | 219 | 不进入 v1 指令集 |

## Train / validation 划分

推荐并已经采用：从可用训练商品内部按二级品类做商品级90/10确定性分层划分，排序键为固定 namespace、seed=42 和商品 ID 的 SHA-256。

- 同一商品的三个 task 永远位于同一 split；
- validation 仅用于训练过程中的 loss、结构和品类表达监控；
- 不额外从训练数据建立 test；
- 冻结的项目 validation/test 不参与模板设计、训练、早停或 P5 spot-check；
- 最终 test 只应在训练方案冻结并获得单独确认后使用。

采用90/10而不是重新做8/1/1，是因为项目已经有独立冻结 test；再从当前训练商品切出 test 会减少有限训练数据，并产生一个职责重叠的新测试集。

## 人工样例审核

每个二级品类抽3件，共24件、72条 task target 完成人工 spot-check，最终为24/24商品、72/72 target 通过。完整样例位于 `lora_instruction_data_v1_samples.csv`，审核记录位于 `lora_instruction_data_v1_spot_check.md`。

## 明显限制与阻断项

### 不阻断 P5 数据构造，但限制效果结论

1. PRD 中“每类品类至少2,000条”属于建议策略；当前每个二级品类只有149–184件商品、447–552条三任务样本，无法达到该建议规模。Leader 9.25 已明确允许先用清洗原标题和结构化属性做轻量 LoRA 验证，因此不阻断工程 smoke，但不能宣称数据覆盖达到 PRD 建议规模。
2. selling points 和 short description 是规则构造的监督文案，不是现成原生优质文案。它们适合学习输出结构和中性表达，但语言多样性有限，不能在训练前声称一定能提升“转化引导性”。
3. 既有一致性审查不是1,559件逐图逐页人工核验；本轮新增规则和24件 spot-check 降低了明显风险，仍可能存在未发现的源属性错误。

### 启动 QLoRA 前的实际阻断项

当前 `ecommerce-generation` 环境已有：Python 3.10.21、torch 2.12.0+cu130、transformers 5.16.1、accelerate 1.14.0、bitsandbytes 0.50.2；缺少 `peft`、`trl`、`datasets`。GPU 为 RTX 4070 Laptop，显存8188MiB。

最小训练实现只需要新增并锁定与当前环境兼容的 `peft`；首个 smoke 可以直接读取 JSONL 并使用 Transformers Trainer/自定义 Dataset，因此不必同时安装 `trl` 和 `datasets`。只有决定使用 `SFTTrainer` 后才增加 `trl` 与 `datasets`。本阶段没有安装任何依赖。

## 后续8GB QLoRA smoke 建议（未执行）

- 基础模型继续固定 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`；
- 4-bit NF4、double quant、bfloat16 compute；调用 `prepare_model_for_kbit_training`；
- 首次使用 `q_proj`、`v_proj`，LoRA rank=8、alpha=32、dropout=0.05；
- micro batch=1、gradient accumulation=8或16、gradient checkpointing 开启、`use_cache=false`；
- 序列上限先设320 token。本地 tokenizer 对4,020条 messages 的统计为 min=102、P50=153、P95=215、P99=231、max=269，320不会截断；
- smoke 只取训练 split 每品类1件（24个 task 样本），另取 LoRA validation 每品类1件作结构检查；只跑2–5个 optimizer step；
- 验证模型4-bit加载、可训练参数数量、一次前向/反向、loss有限、峰值显存、adapter保存与重载；
- 若发生 OOM，先保持 batch=1 并缩短序列或减少 target modules，不直接改变数据版本；
- smoke 不运行项目 test，不评价正式效果，不把 smoke loss 作为模型优劣结论。

Hugging Face 官方文档说明，量化模型可通过 PEFT adapter 继续训练，并推荐4-bit NF4、double quant、bfloat16 compute 及 `prepare_model_for_kbit_training`；TRL 的 SFTTrainer 可直接处理 conversational `messages` 数据，但它不是首个最小 smoke 的必需依赖：[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)、[Transformers bitsandbytes](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)。

## 阶段状态

- 数据构造：完成；
- 24件/72 target spot-check：完成；
- QLoRA 依赖安装：未执行；
- 模型训练：未执行；
- 项目 test：未运行；
- Git commit/push：未执行。
