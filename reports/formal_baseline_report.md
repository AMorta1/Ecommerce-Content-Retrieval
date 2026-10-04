# Week 1 正式 Baseline 归档报告

归档状态：**已完成**  
归档时间（UTC）：`2026-09-23T13:15:28+00:00`  
机器可读清单：`reports/formal_baseline_manifest.json`

本次归档只引用已有冻结数据、配置、输出、人工标签、指标和实验产物，并完成哈希核对；**没有重新运行生成或检索 Baseline**。

## 1. 归档依据与口径

- Leader 最新明确要求：固定当前测试集、评测查询集、各模块 Baseline 指标和模型参数，后续优化只对标这套基准，不再改变评测集和评测口径。
- PRD：生成侧使用固定 test100 并记录人工质量指标；检索侧使用固定 test50，正式门槛为 P@10 和 R@10。
- 已确认实现方案：生成正式版本为 `generation_sampling_prompt_v2`；检索正式版本为 `week1_v3_test50_crossmodal_v3`。
- 仓库现有 MRR@10 和 NDCG@10 已纳入归档，二者属于补充指标，不改写为 PRD 强制门槛。

## 2. 代码与环境版本

```yaml
code_version:
  git_commit: beaf9d1c77719a2b0d1a4968b05444cc357e8e7e
  baseline_commit_short: beaf9d1
  commit_message: 对数据处理和第一周基线的实现进行优化
  branch_at_baseline: Week1
```

完整 commit SHA 是 Baseline 代码版本的主要唯一标识。`Week1` 只记录提交当时的分支。依赖和环境单独归档，不能由 Git commit 替代。

归档时观测到的本机环境：

| 环境 | 关键信息 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB，Driver 581.29，NVIDIA-SMI CUDA 13.0 |
| `ecommerce-generation` | Python 3.10.21；Torch 2.12.0+cu130；Transformers 5.16.1；Accelerate 1.14.0；bitsandbytes 0.50.2 |
| `ecommerce-retrieval` | Python 3.10.21；PaddlePaddle GPU 3.3.0；PaddleNLP 3.0.0b4；NumPy 1.26.4；Faiss CPU 1.15.0；Pillow 12.3.0 |

`requirements-generation.txt`、`requirements-retrieval.txt`、`requirements-data.txt` 及其哈希已写入版本清单。`requirements-data.txt` 声明 Pillow 11.1.0，而归档时检索环境中是 12.3.0；该差异作为环境限制保留，没有修改依赖。

原始结果没有记录两个模型的不可变 revision。当前本机 Qwen 缓存的 `main` ref 为 `a09a35458c702b33eeacc393d103063234e8bc28`，它只代表归档时的本机观察，不能反向证明运行 Baseline 时使用的精确 revision。

## 3. 冻结评测资产

| 资产 | 数量 | SHA256 |
| --- | ---: | --- |
| `data/processed/week1_v3/multimodal/test.jsonl` | 200 | `4afe6589d2f2a6b8d020ec3836128c6a83d00bf6acd3eff0c0b5f0a8a68076a5` |
| `data/processed/week1_v3/generation_evaluation_samples.jsonl` | 100 | `ce057356a9a0c48e4298daac545fafb0c83ad07ae2f74f1b1211c834b048a557` |
| `configs/retrieval_test_queries.jsonl` | 50 | `48253c3287967d90caa5f9a7b3103ca00a1126d401b3cfe067cb96da0261443f` |
| `configs/retrieval_test_query_texts.json` | 50 条查询对应文本 | `2194f9d0c2587390a898a3f2a3f09062c5da6630fd9dde1342b7264617e5ddaa` |
| `reports/retrieval/test_evaluation/query_review.csv` | 50 条查询审核 | `f008e6ffb4c626246ceb119fea4531072a8aaee59a6f6a240af9dc90d669b290` |
| `reports/retrieval/test_evaluation/relevance_judgments.csv` | 1,200 行人工标签 | `dbf7aecb0dd98b18986027c819749dbf9a723f7973a5ba11263b595f63a3b6f3` |

冻结资产不得修改、替换或重新抽样。后续实验必须使用独立文件保存配置、输出、排名、人工标注和指标。

## 4. 生成 Baseline

版本：`generation_sampling_prompt_v2`  
模型：`Qwen/Qwen2.5-7B-Instruct`  
设置：NF4 4-bit；Prompt `baseline_v2`；`do_sample=true`；`temperature=0.7`；`top_p=0.9`；seed 42；最大输入 2048 tokens；最大新生成 320 tokens。

| 指标 | 正式结果 | PRD 门槛 | 结论 |
| --- | ---: | ---: | --- |
| 核心属性命中率 | 88.16% | ≥75% | 达标 |
| 通顺率 | 98.00% | ≥80% | 达标 |
| 平均生成耗时 | 6.42981 秒 | ≤12 秒 | 达标 |
| 事实错误样本率 | 73.00% | 未设门槛 | 后续 RAG 的主要对照 |
| 平均事实错误数 | 1.33 条/样本 | 未设门槛 | 归档指标 |
| 品类风格通过率 | 100.00% | 未设门槛 | 归档指标 |
| 结构成功率 | 99.00% | 未设门槛 | 99/100 成功解析 |
| P95 生成耗时 | 7.707 秒 | 未设门槛 | 归档指标 |
| 最大生成耗时 | 8.798 秒 | 未设门槛 | 归档指标 |

人工标注 100/100 完成，未使用助手草拟标签。旧版 `generation_baseline_v1` 继续保留为历史实验。正式版本相较旧版同时改变 Prompt 和采样方式，因此两版差异不能归因到单一变量。

## 5. 检索 Baseline

版本：`week1_v3_test50_crossmodal_v3`  
模型：`PaddlePaddle/ernie_vil-2.0-base-zh`  
策略：纯跨模态；文本查询检索图像特征，图片查询检索文本特征；Faiss `IndexFlatIP`；L2 归一化后以内积实现余弦相似度；Top-10；相关等级 ≥1 记为相关；查询商品自身从候选库排除。

### 5.1 整体指标

| 查询模式 | P@10 | R@10 | MRR@10 | NDCG@10 | PRD P/R 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 文本搜商品图片 | 43.60% | 61.22% | 0.725270 | 0.569573 | 均未达到 55% / 65% |
| 图片搜商品文本 | 45.40% | 62.68% | 0.694524 | 0.582210 | 均未达到 55% / 65% |

P@10 和 R@10 是 PRD 正式指标；MRR@10 和 NDCG@10 是当前仓库已经计算并保存的补充排序指标。

### 5.2 一级品类指标

| 查询模式 | 一级品类 | P@10 | R@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| 文本搜图 | 3C数码 | 43.20% | 57.52% | 0.738000 | 0.550317 |
| 文本搜图 | 家居日用 | 44.00% | 64.92% | 0.712540 | 0.588829 |
| 图片搜文 | 3C数码 | 44.00% | 59.06% | 0.682381 | 0.553258 |
| 图片搜文 | 家居日用 | 46.80% | 66.29% | 0.706667 | 0.611161 |

二级品类完整指标保存在机器可读清单的 `retrieval_baseline.metrics_by_query_mode.*.by_category_l2` 中，原始逐查询明细保存在正式指标文件中。

50 条查询和 1,200 行相关性标签均已人工确认。validation 的 24 查询候选池结果只保留为历史试跑，不替代本 test50 基准。融合特征和融合索引属于优化候选，不属于本次正式纯跨模态 Baseline。

## 6. 完整性与限制

- 版本清单共登记 36 个现有文件，并对其中 14 个已有报告内置哈希的关键文件完成一致性核对，全部通过。
- 生成侧 100 条输出中有 1 条结构解析失败，原始输出仍完整保留，正式结构成功率为 99%。
- 检索语料的 1,992 条文本特征实际都只使用标题，因为 description 为空。
- 原始 Baseline 报告没有记录不可变模型 revision；本报告没有把当前缓存状态冒充为历史运行时 revision。
- 归档时环境检查发现 Pillow 声明版本与当前检索环境版本不一致，已记录但未改动依赖。

## 7. 后续使用规则

后续版本必须在相同冻结数据和相同指标定义下比较，不得覆盖本报告引用的 Baseline 文件。validation 用于开发和调参；配置锁定后再运行 frozen test；不得根据 test 结果反复改参数后重测。
