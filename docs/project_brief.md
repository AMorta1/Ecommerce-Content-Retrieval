# 电商内容智能生成与多模态检索算法验证项目简报

## 1. 项目背景与目标

项目面向商品上新文案生产与电商商品搜索，验证两类算法能力：基于商品结构化属性生成标题、卖点和短详情；基于图文联合表征实现文本搜图与图片搜文。

采用成熟预训练模型及轻量优化路线，在有限硬件资源下建立可运行、可评测、可追溯的算法原型。项目定位为算法验证，不覆盖推荐、广告或生产级服务系统。

## 2. 数据与任务范围

实验覆盖3C数码、家居日用，以及耳机、键盘、鼠标、移动电源、保温杯、收纳箱、拖把、垃圾桶八个二级品类。

| 数据／评测口径 | 数量与用途 |
|---|---|
| 有效图文商品 | 1,992件 |
| 项目划分 | 初始train 1,592件；project_validation 200件；project_test 200件 |
| 训练侧去重版本 | week2_train_v1，1,559件 |
| LoRA指令数据 | 1,340件商品、4,020条三任务记录 |
| lora_train | 1,206件／3,618任务，仅用于参数更新 |
| lora_train_dev | 134件／402任务，用于训练监控与最佳checkpoint选择 |
| 正式生成人工评测 | 冻结test100，100件商品、625项核心属性分母 |
| 正式检索评测 | 冻结test50、test200候选库、1,200条人工相关性标签 |

LoRA内部90/10划分仅作用于原训练侧，不替代项目级划分。数据依据为[Baseline冻结报告](../reports/formal_baseline_report.md)、[图文处理汇总](../data/processed/week1_v3/multimodal/summary.json)、[训练去重报告](../reports/data/training_deduplication_week2_v1.json)及[LoRA数据final manifest](../reports/generation/lora/lora_instruction_data_v1_final_manifest.json)。字段与剔除规则见[数据处理说明](data.md)。

## 3. 技术方案

### 3.1 商品内容智能生成

采用Qwen2.5-7B-Instruct。Baseline单次调用生成完整文案；Base/LoRA对照按title、selling_points、short_description三个任务分别调用。生成输入为品类和可靠属性，原标题不进入模型输入。

正式LoRA及后续评测固定revision为a09a35458c702b33eeacc393d103063234e8bc28。Base/LoRA对照的输入、Prompt/chat template、解码参数和逐商品逐任务seed保持一致，唯一模型变量为adapter加载。历史Baseline的不可变revision未完整记录，不以当前缓存补证。

### 3.2 RAG增强

采用同商品结构化事实选择与质量门控，不依赖跨商品知识补全。RAG v1使用身份信息、共享Top-3事实和否定约束；四区机制进一步区分Identity、Mandatory Reliable Core、Supplemental和Negative Constraints，使可靠核心事实不受补充Top-3名额限制。

对Mandatory事实执行tokenizer后完整性检查，超限或缺失直接报错。字段缺失、placeholder及源冲突按policy处理，不修改正式核心属性分母。组合方案在冻结LoRA上增加四区上下文及必要输入接口适配，无新增训练。

### 3.3 LoRA / QLoRA

基于清洗后的原生标题与可靠属性构造三任务target，排除已确认冲突、暂缓待审商品，不使用模型生成结果作为训练真值。训练目标为电商表达结构和品类话术适配。

采用NF4 4-bit、double quant和BF16，LoRA注入q_proj/v_proj，rank=8、alpha=32。通过梯度累积、gradient checkpointing及assistant-only分块loss完成8GB显存下的训练。完整训练3 epochs，总耗时99.12分钟；最佳模型按完整lora_train_dev最低loss选择，为epoch 3 / checkpoint-step-001359。来源：[正式训练报告](../reports/generation/lora/lora_v1_training/final_training_report.md)。

### 3.4 图文跨模态商品检索

采用PaddlePaddle/ernie_vil-2.0-base-zh分别编码图像和文本，输出768维特征，经L2归一化后使用Faiss IndexFlatIP进行余弦相似度检索。文本搜图与图片搜文使用不同模态索引；正式评测排除查询商品自身。

原生短描述为空，实际商品文本特征仅使用标题。模型、维度与特征策略来源为[Baseline冻结manifest](../reports/formal_baseline_manifest.json)；精确特征接口与预处理记录见[Retrieval技术说明](retrieval.md)。

### 3.5 Retrieval Rerank

文本搜图在Top-20候选中结合品类、属性及数值约束重排，再输出Top-10。冻结公式为相似度乘品类因子，加0.24倍约束命中比例，减0.12倍冲突比例。

图片查询缺少独立可靠的结构化信号，图片搜文保持Baseline，不使用查询商品隐藏属性。参数依据为[rerank冻结manifest](../reports/retrieval/rerank/rerank_v1_formal_manifest.json)；详细技术实现见[generation](generation.md)、[retrieval](retrieval.md)和[evaluation](evaluation.md)。

## 4. 关键实验结果

本节以最终正式报告为事实来源。单调用Baseline/RAG、三任务Base/LoRA和追加组合分别解释，不将跨组差异归因于单一模型变量。核心命中采用人工语义复核；自动字面覆盖及Grounding仅作辅助。

### 4.1 Generation Baseline

采用冻结generation_sampling_prompt_v2，正式test100结果如下：

| 核心属性命中率 | 通顺率 | 事实错误样本率 | 平均事实错误数／件 | 平均模型生成耗时 |
|---:|---:|---:|---:|---:|
| 551/625，88.16% | 98.00% | 73.00% | 1.33 | 6.42981秒 |

Baseline达到核心覆盖和通顺要求，但存在较多无依据事实扩写。来源：[正式Baseline报告](../reports/formal_baseline_report.md)。

### 4.2 RAG

最终RAG v1正式结果采用formal test100最终人工复核，不使用早期复核草稿：

| 版本 | 核心属性命中率 | 通顺率 | 事实错误样本率 | 平均事实错误数／件 | 平均模型生成耗时 |
|---|---:|---:|---:|---:|---:|
| RAG v1 | 405/625，64.80% | 93.00% | 35.00% | 0.57 | 4.344秒 |

相对单调用Baseline，事实错误样本率下降38.00个百分点，但核心覆盖下降23.36个百分点，未达到75%要求。来源：[RAG v1最终报告](../reports/generation/rag/formal_rag_v1_test100_report.md)。

RAG v2只有已归档的development24诊断结果，没有正式test结果：

| 开发版本 | 核心属性命中率 | 通顺率 | 事实错误样本率 | 平均事实错误数／件 |
|---|---:|---:|---:|---:|
| 同批V1-control | 78.36% | 91.67% | 45.83% | 0.71 |
| V2-context | 89.55% | 100.00% | 70.83% | 2.71 |
| 最终受控事实候选 | 87.31% | 37.50% | 33.33% | 1.08 |

四区机制恢复了核心属性覆盖，但未同时满足事实性与表达要求。最终受控候选通顺率未达门槛，原RAG v2未进入正式评测，32件generation-output holdout保持frozen_not_executed。以上是[最终阶段归档](../reports/generation/rag/rag_v2_p2_phase_summary.md)中的开发结论，不替代test100指标。

### 4.3 LoRA

采用冻结epoch 3 adapter，在相同三任务推理条件下进行Base/LoRA正式test100对照：

| 指标 | Base | LoRA v1 |
|---|---:|---:|
| 核心属性命中率 | 556/625，88.96% | 467/625，74.72% |
| 通顺率 | 98.00% | 99.00% |
| 事实错误样本率 | 92.00% | 1.00% |
| 平均事实错误数／件 | 3.72 | 0.05 |
| 三任务平均模型生成耗时 | 4.899秒 | 4.994秒 |

在本样本及冻结人工口径下，LoRA显著减少无依据事实扩写，但核心覆盖降低14.24个百分点，低于75%要求。该覆盖下降不能仅归因于训练数据；输入门控与输出遗漏的区分见正式分析。

| 匿名表达比较维度 | Base更好 | LoRA更好 | 持平 |
|---|---:|---:|---:|
| 标题表达质量 | 93 | 6 | 1 |
| 卖点结构 | 7 | 93 | 0 |
| 短详情自然度 | 98 | 2 | 0 |
| 整体电商专业度 | 8 | 91 | 1 |

每项为100组配对偏好，不是质量通过率。LoRA改善卖点结构与整体规范性，但标题及短详情自然度仍不足。来源：[LoRA最终test分析](../reports/generation/lora/project_test100_v1/final_test_analysis.md)。

### 4.4 LoRA + RAG

组合复用同一LoRA adapter，并引入RAG v2四区事实机制。在原固定test100上追加对照：

| 指标 | LoRA 对照组 | LoRA + RAG |
|---|---:|---:|
| 核心属性命中率 | 74.72% | 73.76% |
| 通顺率 | 99.00% | 97.00% |
| 事实错误样本率 | 2.00% | 1.00% |
| 平均事实错误数／件 | 0.05 | 0.01 |
| 三任务平均模型生成耗时 | 4.994秒 | 5.554秒 |

整体专业度偏好为LoRA 17、组合13、持平70；卖点结构95组持平，短详情自然度99组持平。组合减少了少量事实错误，但未表现出额外属性覆盖或稳定整体表达收益。来源：[组合最终复核报告](../reports/generation/lora_rag/lora_rag_project_test100_v1/final_review_report.md)。

同批 LoRA 对照复用冻结LoRA文本、重新配对复核，错误样本率2%与独立LoRA评测的1%属于不同冻结标签，均保留原报告，不相互覆盖。LoRA 对照组耗时为历史记录，非同期性能对照。四区上下文和system接口适配为联合处理，不作纯检索单因素归因。

### 4.5 Retrieval

采用冻结week1_v3_test50_crossmodal_v3及retrieval_rerank_v1_formal_test50：

| 方向／版本 | P@10 | R@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Text→Image Baseline | 43.60% | 61.22% | 0.725270 | 0.569573 |
| Text→Image Rerank v1 | 51.80% | 71.96% | 0.804524 | 0.730356 |
| Image→Text Baseline／Rerank | 45.40% | 62.68% | 0.694524 | 0.582210 |

规则重排改善文本搜图总体排序，Recall达到65%目标，Precision仍低于55%；图片搜文保持原结果，两个门槛均未达到。各品类收益不一致，不能将总体改善推广为所有类别均提升。来源：[检索正式rerank报告](../reports/retrieval/rerank/formal_rerank_v1_test50_report.md)。

## 5. PRD指标完成情况

| 要求 | 当前结果与判断 |
|---|---|
| 生成核心覆盖≥75% | Baseline 88.16%，达到；RAG v1 64.80%、LoRA v1 74.72%、组合73.76%，未达到 |
| 生成通顺率≥80% | 正式Baseline/RAG v1/LoRA/组合分别98%/93%/99%/97%，均达到；原RAG v2未晋级 |
| 单条生成响应≤12秒 | 已记录模型生成均值6.42981/4.344/4.994/5.554秒，均低于12秒；不等同完整服务端响应验收 |
| 检索P@10≥55% | 文搜图51.80%、图搜文45.40%，未达到 |
| 检索R@10≥65% | 文搜图71.96%，达到；图搜文62.68%，未达到 |
| 单次检索响应≤6秒 | 已记录查询编码、Faiss搜索及批量评测耗时；尚缺完整单次检索链路的正式计时验收，暂不判定达标 |

门槛依据项目PRD非功能需求及正式Baseline报告。模型计时不包含完整业务链路，不将局部计时等同全部性能要求已通过。

## 6. 项目成果

- 建立商品清洗、字段标准化、图片校验、训练侧去重与数据隔离流程。
- 完成生成Baseline、事实增强、QLoRA与组合方案的实现及冻结对照。
- 形成8GB设备下可训练、可恢复的QLoRA工程方案与最终adapter。
- 完成双向跨模态特征库、Faiss检索及文本侧冻结rerank。
- 建立人工核心覆盖、通顺、事实性、匿名表达偏好及检索指标体系。
- 整理数据、技术、评测和交付资料，保留配置、输出、标签、代码快照及hash关系。

## 7. 当前限制

正式test100在多个阶段被观察，追加组合结果不是新的未观察盲测。小规模结果用于本项目样本描述，不作总体性能或普适幻觉率估计。

源属性仍有缺失、多SKU混写和身份冲突，未逐项通过外部商品页面核实；LoRA模板化target与当前生成方式在表达自然度和信息覆盖之间存在权衡，确切影响需新实验才能确认。

历史Baseline／ERNIE-ViL不可变revision与部分环境锁定证据不完整；通用LoRA/组合推理CLI、在线rerank及Demo仍存在工程缺口。测试阶段保持模型、Prompt、数据及评测协议冻结，未基于测试集结果继续训练或调整参数。

## 8. 交付内容

| 内容 | 交付位置／方式 |
|---|---|
| 代码、配置、测试与使用入口 | src、scripts、configs、tests及[README](../README.md) |
| 数据处理与技术说明 | [data](data.md)、[generation](generation.md)、[retrieval](retrieval.md)、[evaluation](evaluation.md) |
| 正式实验与追溯材料 | 本简报各结果节所链接的报告、final manifest、人工标签、环境与checksums |
| 最终adapter | artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter |
| 检索特征、索引与商品映射 | artifacts/features、artifacts/indexes |
| 基座与环境 | 仓库外固定模型snapshot及对应版本／环境记录 |
| 交付与历史材料索引 | [delivery audit](delivery_audit.md)，历史实验原位保留 |

