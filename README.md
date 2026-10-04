# Ecommerce-Content-Retrieval

## 1. 项目简介

电商内容智能生成与多模态检索算法验证项目，包含两个核心模块：

- 商品内容智能生成：以品类和可靠结构化属性为输入，使用Qwen2.5-7B-Instruct生成标题、卖点和短详情，提供Baseline、RAG、LoRA及组合方案的实现与评测证据。
- 图文跨模态商品检索：使用ERNIE-ViL中文图文编码与Faiss，实现文本搜图、图片搜文及文本侧规则重排。

本仓库提供算法代码、数据处理链、配置、模型附件和实验报告。项目成果见[项目简报](docs/project_brief.md)，使用与技术说明见下文索引。

## 2. 项目结构

~~~text
src/data/          数据解析、清洗、图片校验与去重
src/generation/    Qwen、RAG、LoRA数据/训练/推理与评测实现
src/retrieval/     ERNIE-ViL、特征库、Faiss、rerank与评测
scripts/           命令行入口
configs/           数据、生成、训练、检索和评测配置
docs/              项目简报、技术说明、评测协议与交付审计
reports/           正式报告、指标、人工复核、manifest与历史证据
artifacts/         LoRA adapter/checkpoint、特征、索引及商品映射
tests/             数据规则、算法与工程保护测试
data/              处理后数据及图片附件，不随Git提供
~~~

正式入口与历史工具的分类见[交付审计](docs/delivery_audit.md)。不要将scripts目录中的所有脚本依次运行。

## 3. 环境说明

生成与检索使用独立环境。以下是已记录的本机版本，不是对任意平台的兼容性保证。

| 环境 | 关键依赖 |
|---|---|
| Generation：ecommerce-generation | Python3.10.21；Torch2.12.0+cu130；Transformers5.16.1；Accelerate1.14.0；bitsandbytes0.50.2；PEFT0.21.1 |
| Retrieval：ecommerce-retrieval | Python3.10.21；Paddle GPU3.3.0；PaddleNLP3.0.0b4；NumPy1.26.4；Faiss CPU1.15.0 |
| 数据处理／工作簿 | Pillow；openpyxl，具体环境差异见技术文档 |

依赖声明：[Generation](requirements-generation.txt)、[Retrieval](requirements-retrieval.txt)、[图片处理](requirements-data.txt)。Generation声明尚未包含PEFT，数据声明与历史检索环境的Pillow版本也存在差异；复现应同时参考[正式训练环境快照](reports/generation/lora/lora_v1_training/environment_20261004T064046462933.txt)和[Baseline环境记录](reports/formal_baseline_report.md)，不能仅把requirements当作完整锁文件。

已有环境先检查版本，不默认升级核心包：

~~~powershell
conda activate ecommerce-generation
python --version
python -m pip check
~~~

检索环境同样先激活ecommerce-retrieval再检查。基座缓存位于仓库外../.model-cache；Generation正式流程使用固定本地snapshot。详细加载与环境说明见[generation](docs/generation.md)和[retrieval](docs/retrieval.md)。

## 4. 数据说明

图文有效数据共1,992件，覆盖3C数码、家居日用及八个二级品类。

| 名称 | 数量 | 用途 |
|---|---:|---|
| 项目初始train | 1,592件 | 原训练侧 |
| project_validation | 200件 | 项目级开发和候选验证 |
| project_test | 200件 | 冻结正式测试侧 |
| lora_train | 1,206件／3,618任务 | LoRA参数更新 |
| lora_train_dev | 134件／402任务 | dev loss监控和checkpoint选择 |

LoRA内部90/10来自原训练侧，不替代项目级划分。正式生成人工样本为固定test100，正式检索为固定test50。路径、字段、清洗与剔除规则见[数据处理说明](docs/data.md)。未交付数据/图片附件时，不能完整运行数据相关流程。

## 5. 快速使用

以下命令均在仓库根目录执行，供非正式使用。已完成的正式训练、test和holdout不得重跑或覆盖。

### Generation Baseline

输入必须含product_id、title和generation_input，推荐使用已有非正式推理样例。为避免覆盖，先检查输出路径：

~~~powershell
conda activate ecommerce-generation
if (Test-Path -LiteralPath 'reports/user_demo/baseline.json') { throw '输出已存在，请选择新路径' }
python -B -X utf8 scripts/generate.py --config configs/generation.json --input data/processed/week1_v3/inference_samples.jsonl --sample-count 1 --offline --output reports/user_demo/baseline.json
~~~

这是单调用Baseline演示，不是正式test或LoRA三任务对照。

### LoRA、RAG与LoRA + RAG

| 功能 | 推荐实现／正式入口 | 使用边界 |
|---|---|---|
| LoRA训练 | [train_lora.py](scripts/train_lora.py)，[训练config](configs/lora_qlora_train_v1.json) | 训练已完成；仅提供入口定位 |
| LoRA推理 | [lora_validation.py](src/generation/lora_validation.py)中的generate加载链；正式test入口为[run_lora_project_test.py](scripts/run_lora_project_test.py) | 从固定base加载最终adapter；通用单商品CLI尚未提供 |
| RAG v1生成 | [run_rag_formal_test.py](scripts/run_rag_formal_test.py)，[formal config](configs/generation_rag_formal_v1.json) | 已封存的一次性test入口，不作为演示重跑 |
| LoRA + RAG | [run_lora_rag_project_test.py](scripts/run_lora_rag_project_test.py)，[组合config](configs/lora_rag_project_test100_v1.json) | 已封存的组合对照入口；通用单商品CLI尚未提供 |

最终adapter：[checkpoint-step-001359/adapter](artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter)。加载链、输入要求和任务模板见[generation技术说明](docs/generation.md)。这些正式评测脚本不是通用服务入口，不能仅更换输入路径就用于任意商品。

### ERNIE-ViL与检索

准备已有模型缓存、特征、索引和商品映射后，可进行非正式检索：

~~~powershell
conda activate ecommerce-retrieval
$env:PPNLP_HOME = (Resolve-Path '../.model-cache/paddlenlp').Path
python -B -X utf8 scripts/search.py --text "USB接口的迷你有线键盘" --top-k 5 --json
python -B -X utf8 scripts/search.py --image data/images/week1_v3/16454614360.webp --top-k 5 --json
~~~

search.py是跨模态Baseline入口，尚未集成正式rerank。

| 功能 | 正式入口 |
|---|---|
| 图像embedding及索引 | [build_index.py](scripts/build_index.py)，[retrieval.json](configs/retrieval.json) |
| 文本embedding及索引 | [build_text_features.py](scripts/build_text_features.py) |
| Retrieval evaluation | [evaluate_test_retrieval.py](scripts/evaluate_test_retrieval.py)，[test evaluation config](configs/retrieval_test_evaluation.json) |
| Rerank evaluation | [run_retrieval_rerank_formal_test.py](scripts/run_retrieval_rerank_formal_test.py)，[rerank config](configs/retrieval_rerank_formal_v1.json) |
| Generation evaluation | [evaluate_generation.py](scripts/evaluate_generation.py)，具体人工口径见[evaluation](docs/evaluation.md) |

上述构建/评测入口用于实现定位；冻结特征、索引和正式结果已有附件，不应重建或重测已有冻结结果。

## 6. 配置文件

主要配置均位于configs：

- Generation Baseline：generation.json。
- LoRA数据／训练：lora_instruction_data_v1.json、lora_qlora_train_v1.json。
- 三任务推理／正式测试：lora_project_validation_v1.json、lora_project_test100_v1.json。
- RAG：generation_rag_formal_v1.json、rag_fact_policy_v3.json；四区事实及组合使用rag_fact_policy_v4.json。
- 组合正式测试：lora_rag_project_test100_v1.json。
- 检索／重排：retrieval.json、retrieval_rerank_formal_v1.json。
- 评测定义：generation_evaluation.json、retrieval_test_evaluation.json。

配置中的历史pending状态不代表当前实验未完成，最终状态以对应final manifest为准。冻结配置须保持不变。

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| [项目简报](docs/project_brief.md) | 项目方案、主要结果、目标完成情况与交付成果 |
| [数据处理说明](docs/data.md) | 来源、品类、字段、清洗、剔除与数据隔离 |
| [Generation技术说明](docs/generation.md) | Qwen、RAG、QLoRA、组合与推理实现 |
| [Retrieval技术说明](docs/retrieval.md) | ERNIE-ViL、预处理、特征、索引与rerank |
| [Evaluation说明](docs/evaluation.md) | 人工指标、匿名偏好、检索指标与冻结原则 |
| [Delivery audit](docs/delivery_audit.md) | 正式入口、历史分类、交付缺口与清理记录 |

## 8. 主要结果

以下是各版本的最终正式结果摘要，不构成跨实验条件的统一排行榜。

| 生成版本 | 核心属性命中率 | 通顺率 | 事实错误样本率 |
|---|---:|---:|---:|
| 单调用Baseline sampling v2 | 88.16% | 98% | 73% |
| LoRA v1，独立Base/LoRA对照 | 74.72% | 99% | 1% |
| LoRA + RAG，追加配对对照 | 73.76% | 97% | 1% |

冻结rerank的Text→Image P@10/R@10为51.80%/71.96%；Image→Text保持Baseline，为45.40%/62.68%。完整对照、RAG结果、匿名偏好和PRD判断见[项目简报](docs/project_brief.md)及其引用的正式报告。

## 9. 已知限制

- LoRA及组合的核心覆盖未达到75%；标题与短详情自然度仍有限制，组合未表现出稳定整体表达收益。
- 文搜图Precision和图片搜文P/R未达PRD门槛；检索完整端到端性能验收尚无可靠记录。
- 通用单商品LoRA/组合推理CLI、在线rerank和Gradio Demo尚未在仓库中找到实现。
- 历史Baseline模型revision、部分环境锁定及源数据授权记录不完整。
- test100已在多轮评测中被观察。

## 10. 交付说明

- artifacts已加入Git跟踪并暂存，包括最终adapter、特征、索引和历史工程产物；.gitattributes禁止换行转换以保持附件hash。尚未commit/push，交付前需确认接收方实际获得这些文件。
- data/processed和data/images仍被Git忽略，需提供独立数据/图片附件；模型基座缓存不在仓库内，需另附固定snapshot或固定版本获取说明。
- ARCHIVE原位保留；历史smoke adapter不是正式模型初始化权重。历史分类与缓存清理记录见delivery audit，清理前inventory不是当前Git状态清单。
- 最终报告位于reports/formal_baseline_report.md、reports/generation/{rag,lora,lora_rag}/及reports/retrieval/rerank/，从[项目简报](docs/project_brief.md)进入对应报告和manifest。
