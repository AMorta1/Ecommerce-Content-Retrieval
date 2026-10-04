# RAG v2 P2.6 controlled factual generation 报告

## 结论

P2.6 已按约束完成，但 **development gate 未通过**，因此停止，不进入 P3，不运行32条 generation-output holdout，也不创建 P2.7/P2.8 Prompt。

唯一的新候选 `rag_v2_controlled_factual_v1` 保留了较高的核心属性覆盖，并把事实错误样本率降到 V1-control 以下；但模型在大量短详情中原样抄写 Prompt 给出的“为、是、采用、使用、配备、具有、以及、其中、并”等中性连接词示例，人工通顺率只有37.50%，未达到80%门槛。

## P2/P2.5 checkpoint

- 冻结 manifest：`rag_v2_p25_checkpoint_manifest_v1.json`
- manifest SHA-256：`fd48b5b1670b257916904fe0206ddefb185d9c70d630be1dcea77309cb8562d8`
- Git HEAD：`c29b9f76da6cdedc9d66d27e9df02ea3ed8b1f09`
- checkpoint 记录了当前代码、配置、事实数据、P1/P2/P2.5 报告、development24、人工复核和 holdout 清单的逐文件大小与 SHA-256。
- 未 commit、未 push；既有 P2/P2.5 结果未被覆盖或修改。

## 实现边界

只新增一个内容 Prompt：

- 变体：`V2-controlled-factual`
- Prompt：`rag_v2_controlled_factual_v1`
- 配置：`configs/generation_rag_validation_v8_p26.json`

以下内容保持不变：

- `rag_fact_policy_v4` 及其 SHA-256；
- Identity / Mandatory / Supplemental Top-3 / Negative 四区上下文；
- 模型 revision `a09a35458c702b33eeacc393d103063234e8bc28`；
- NF4、解码参数、最大 token、逐商品 seed；
- grounding validator 与正式人工评测口径；
- 冻结32条 holdout 的 ID、规则、seed 和 manifest hash。

## 测试与运行

- Prompt/配置专项及 generation 测试：59 passed。
- 全仓测试：126 passed。
- smoke8：只作工程检查，16/16 JSON 可解析、26/26 Mandatory 注入、无截断、8/8 seed 配对、holdout overlap=0。
- smoke 后未查看结果调 Prompt，Prompt 内容没有再次修改。
- development24：固定 Prompt 后只运行一次；48/48 JSON 可解析、76/76 Mandatory 注入、无截断、24/24 seed 配对、holdout overlap=0。
- 新候选平均生成耗时5.085秒，P95 6.915秒，最大7.713秒。

第一次用 base 环境启动 smoke 时在模型加载前因缺少 `transformers` 停止，未生成输出。随后 `conda run` 内的子进程完成 smoke 并写出有效报告，但 conda 包装层在回显日志时发生 GBK 编码错误；这没有影响 smoke artifact。development24 使用环境内 Python 直接执行并正常退出。

## 人工评测

P2.6 的 V1-control 24条 raw/parsed outputs 与 P2 development24 逐字一致，因此沿用已经最终复核的 P2.5 V1-control 人工结果；新候选24条重新逐条人工复核。

| 版本 | 核心属性命中率 | 通顺率 | 事实错误样本率 | 事实错误总数 | 平均事实错误数 | identity/category error |
|---|---:|---:|---:|---:|---:|---:|
| V1-control | 105/134，78.36% | 22/24，91.67% | 11/24，45.83% | 17 | 0.71 | 0 |
| P2.5 V2-context | 120/134，89.55% | 24/24，100% | 17/24，70.83% | 65 | 2.71 | 0 |
| P2.5 V2-coverage | 119/134，88.81% | 24/24，100% | 17/24，70.83% | 57 | 2.38 | 1 |
| V2-controlled-factual | 117/134，87.31% | 9/24，37.50% | 8/24，33.33% | 26 | 1.08 | 0 |

新候选的事实错误总数相对 P2.5 V2-context 下降60.00%，相对 V2-coverage 下降54.39%。平均事实错误数仍高于 V1-control，但它不是本轮硬门槛；主要事实门槛采用事实错误样本率。

逐条复核见 `rag_v2_p26_controlled_factual_human_review.csv`。自动 Grounding 为18/24 PASS，只用于辅助，没有替代人工事实错误率。

## 按品类结果

| 二级品类 | 核心属性命中 | 通顺样本 | 有事实错误样本 | 事实错误总数 |
|---|---:|---:|---:|---:|
| 保温杯 | 18/20 | 0/3 | 1/3 | 2 |
| 垃圾桶 | 17/19 | 1/3 | 0/3 | 0 |
| 拖把 | 13/13 | 3/3 | 3/3 | 15 |
| 收纳箱 | 11/15 | 1/3 | 0/3 | 0 |
| 移动电源 | 18/18 | 0/3 | 0/3 | 0 |
| 耳机 | 11/13 | 1/3 | 1/3 | 2 |
| 键盘 | 15/21 | 2/3 | 2/3 | 5 |
| 鼠标 | 14/15 | 1/3 | 1/3 | 2 |

## Development gate

| 条件 | 结果 | 是否通过 |
|---|---:|---:|
| core attribute hit ≥85% | 87.31% | 是 |
| fluency ≥80% | 37.50% | **否** |
| factual-error sample rate ≤ V1-control 45.83% | 33.33% | 是 |
| Mandatory 注入完整率100% | 76/76 | 是 |
| JSON 24/24 | 24/24 | 是 |
| identity/category error=0 | 0 | 是 |

总体 gate：**失败**。

按照已确认边界，P2.6 到此停止：不修改本 Prompt，不继续创建新 Prompt 版本，不进入32条 holdout。后续如继续 RAG v2，应另行评估当前7B模型的自由生成方式或输出结构，而不是继续针对 development24 调 Prompt。
