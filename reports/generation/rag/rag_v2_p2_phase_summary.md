# RAG v2 P1–P2.6 阶段归档

## 阶段结论

P1、P2、P2.5 和 P2.6 已归档。P2.6 未通过 development gate，当前状态固定为：不进入 P3、不运行32条 generation-output holdout、不继续创建 P2.7/P2.8 Prompt，也不运行 formal test100。

## Development24 统一人工指标

四个版本均使用相同的24件商品和冻结的134个核心属性分母。`factual-error sample` 指 `factual_error_count > 0` 的商品；自动 Grounding 只作辅助，不替代人工结论。

| 版本 | Prompt | 核心属性命中率 | 通顺率 | 事实错误样本率 | 事实错误总数 | 平均事实错误数 | identity/category error |
|---|---|---:|---:|---:|---:|---:|---:|
| V1-control | `rag_v6` | 105/134，78.36% | 22/24，91.67% | 11/24，45.83% | 17 | 0.71 | 0 |
| V2-context | `rag_v2_context_v1` | 120/134，89.55% | 24/24，100% | 17/24，70.83% | 65 | 2.71 | 0 |
| V2-coverage | `rag_v2_coverage_v1` | 119/134，88.81% | 24/24，100% | 17/24，70.83% | 57 | 2.38 | 1 |
| P2.6 V2-controlled-factual | `rag_v2_controlled_factual_v1` | 117/134，87.31% | 9/24，37.50% | 8/24，33.33% | 26 | 1.08 | 0 |

## 已验证的结论

### Mandatory Core 对属性覆盖的贡献

在同一批配对商品上，V2-context 相对 V1-control 将核心属性命中率从78.36%提高到89.55%，增加11.19个百分点；V2-coverage 为88.81%，P2.6 为87.31%。这证明把源数据中实际存在且通过质量门控的核心事实作为独立 Mandatory Core 注入，能够恢复并提高核心属性覆盖。

强覆盖措辞没有在 V2-context 基础上继续提高命中率：V2-coverage 反而低0.75个百分点。因此，本阶段证据更支持“提供可靠核心事实本身”是覆盖提升的主要来源，不能把提升归因于更强的覆盖命令。

### 自由生成导致事实扩写

V2-context 和 V2-coverage 虽然获得了更多可靠事实，但自由生成会围绕这些事实继续推导评价、效果、性能、体验和具体使用场景。两者的事实错误样本率均为70.83%，高于 V1-control 的45.83%；事实错误总数分别为65和57，显著高于 V1-control 的17。

这说明“输入事实真实”并不自动保证“生成内容只包含真实事实”。Mandatory Core 解决的是事实供给与覆盖问题，不解决模型围绕事实进行营销式扩写的问题。

### Controlled factual Prompt 的改善与剩余风险

P2.6 把事实错误样本率降至33.33%，比 V1-control 低12.50个百分点；事实错误总数相对 V2-context 下降60.00%，相对 V2-coverage 下降54.39%。这证明限制评价、效果、因果和场景扩写是有效方向。

但 P2.6 仍有26个事实错误、平均1.08个，均高于 V1-control 的17和0.71。剩余错误集中在8个样本中，因此不能仅凭错误样本率降低就认为事实可靠性问题已经解决。

### 语言约束与通顺性的 trade-off

P2.6 通顺率只有37.50%，远低于80%门槛。主要原因是模型机械复制 Prompt 中列举的中性连接词，形成不自然的短详情。该结果表明：对7B自由生成模型施加过细、过显式的语言级限制，虽然可以抑制部分事实扩写，却可能让模型从“自由发挥”转为“机械复述约束文本”，造成明显的表达质量损失。

## 版本与哈希关系

完整的逐文件关系、Prompt 可复算指纹、P2.5 checkpoint 和 P2.6 最终工作区代码快照见 `rag_v2_p2_phase_archive_v1.json`。

### 关键配置与 policy

| 阶段 | 文件/版本 | SHA-256 |
|---|---|---|
| P1 | `rag_fact_policy_v4.json` | `b032bf8835a4bfa11338a6bfddba05392aaa896c103473aa402b4ead25029fea` |
| P2 | `generation_rag_validation_v7.json` | `4b061de967ab2bd6a5b9fd73e3da010d57850503068869c4bcf986de075e2356` |
| P2.6 | `generation_rag_validation_v8_p26.json` | `61183df29d03628b9c6715ad983589359c203cd93ee17f2080d41e27d753eeaf` |
| 全阶段 | 冻结评测定义 `generation_evaluation.json` | `66e97e3d620c83584f2a9bff1757127117043dfe154e4dcbfbfd40cccc7e00bd` |

### Prompt 指纹

指纹是固定 canonical fixture 经过当前 `build_rag_messages` 渲染后，对稳定 UTF-8 JSON 序列化结果计算的 SHA-256。它用于区分 Prompt 行为版本，不替代源码文件哈希。

| Prompt 版本 | Canonical rendered SHA-256 |
|---|---|
| `rag_v6` | `22d4eb4e75fbdea6cebaa3c9a816245736f1ed05350e8476d0285dedbaca20d4` |
| `rag_v2_context_v1` | `cc008a24903009b78ed543f2e7a947dbe7983ac469f588003a27a3c3264a3eeb` |
| `rag_v2_coverage_v1` | `0fbf539554e6d08adb39a9c1773663d4425218ea0d873b8cd9b25e0af3b6d765` |
| `rag_v2_controlled_factual_v1` | `a265d6b7141f7becc27b047af70a31ecb2ac288b5c545e813933a62862588cc7` |

### 代码快照

- P1/P2/P2.5 冻结点：`rag_v2_p25_checkpoint_manifest_v1.json`，SHA-256 为 `fd48b5b1670b257916904fe0206ddefb185d9c70d630be1dcea77309cb8562d8`。
- P2.6 最终 `qwen.py`：`344f4e41de9308bf7d213c2786e6022917943f21434c5615239312adf4214aad`。
- P2.6 最终 `run_rag_validation.py`：`168c1f29e144535afa6be762969eda5c456abbdb2b6d5bf586ca59e87d1dc611`。
- P2.6 最终 `test_generation.py`：`6caea6d27b684d8f7292605e2d990d4c8d0aff84880445bde3aa3bac8b444d3b`。
- `build_rag_facts.py`、`grounding.py` 和 `rag.py` 在 P2.6 中未变化，其哈希与 P2.5 checkpoint 一致。

## 32条 holdout 状态

- Manifest：`rag_v2_generation_output_holdout_v1.json`
- SHA-256：`31fae8cbfc53e0f5ef3329cbee6812963b36ddf48e61858d8d9f65fa17ff8da1`
- 商品 ID 清单 SHA-256：`fcf23fdff25cc8c0c6efb46bb1fd254687f652e47c64bca058132573a87a44d5`
- 当前状态：`frozen_not_executed`
- `v2_generation_executed=false`
- 本阶段未读取其生成结果，因为不存在任何生成结果；未运行 V1-control 或任何 V2 候选。

## 建议提交 Git 的文件范围

以下只是建议范围，本次不执行 commit 或 push。

### 1. P1/P2 实现、配置与测试

- `scripts/build_rag_facts.py`
- `scripts/run_rag_validation.py`
- `src/generation/grounding.py`
- `src/generation/qwen.py`
- `src/generation/rag.py`
- `tests/test_generation.py`
- `configs/rag_fact_policy_v4.json`
- `configs/generation_rag_validation_v7.json`
- `configs/generation_rag_validation_v8_p26.json`

### 2. P1/P2 工程归档

- `reports/generation/rag/rag_v2_static_audit_v1.json`
- `reports/generation/rag/rag_v2_validation_fact_build_v1.json`
- `reports/generation/rag/rag_v2_generation_output_holdout_v1.json`
- `reports/generation/rag/rag_v2_p2_smoke8*.json`
- `reports/generation/rag/rag_v2_p2_development24.json`
- `reports/generation/rag/rag_v2_p2_summary_v1.json`

### 3. P2.5 人工诊断与 checkpoint

- `reports/generation/rag/rag_v2_p25_development24_human_review.csv`
- `reports/generation/rag/rag_v2_p25_development24_review_workbook.xlsx`
- `reports/generation/rag/rag_v2_p25_development24_report.md`
- `reports/generation/rag/rag_v2_p25_checkpoint_manifest_v1.json`

### 4. P2.6 与阶段总结

- `reports/generation/rag/rag_v2_p26_smoke8.json`
- `reports/generation/rag/rag_v2_p26_development24.json`
- `reports/generation/rag/rag_v2_p26_controlled_factual_human_review.csv`
- `reports/generation/rag/rag_v2_p26_summary_v1.json`
- `reports/generation/rag/rag_v2_p26_report.md`
- `reports/generation/rag/rag_v2_p2_phase_archive_v1.json`
- `reports/generation/rag/rag_v2_p2_phase_summary.md`

`data/processed/week2_rag_mandatory_core_v2/validation_fact_units.jsonl` 当前受 `.gitignore` 的 `/data/processed/` 规则管理，不建议为了本阶段归档强制加入 Git；其 SHA-256 已记录为 `956cbb567d11c9d603d34f875895f64baa3bdabe584ba185de028f5a7869a798`，并可由已归档代码、policy 和源数据重建。

RAG v1 formal 文件、旧历史版本和32条 holdout 的任何生成输出均不在本阶段建议提交范围内。

## 后续方向建议（仅建议，不执行）

### 将重点转向 LoRA

建议将后续研究重点转向“RAG 负责提供和约束事实，LoRA 负责稳定表达”。本阶段已经证明事实供给与语言表达是两个不同问题：Mandatory Core 可以恢复覆盖，但仅靠自由生成 Prompt 很难同时控制事实扩写和通顺性。LoRA 更适合学习固定输出结构、中性电商表达和自然连接方式，但前提是训练目标本身经过事实一致性审查，不能把营销扩写或当前模型错误当成监督答案。

这只是方向建议，不代表现在启动数据构造、安装依赖或训练。

### 如果未来继续 RAG 生成控制

应建立新的结构化生成实验版本，而不是继续在当前 development24 上追加 Prompt 调参。可研究的边界是把“事实选择”和“语言实现”拆开，例如先输出受 schema 约束的字段槽位，再使用确定性或受控实现层组织文本，并为新版本重新定义开发数据、冻结条件和独立验证流程。

当前 development24 应视为已经充分用于 P2 选择和诊断的数据，不再承担后续 Prompt 迭代验证职责。

## 最终状态

P2 阶段归档完成。当前没有新的 Prompt 候选，没有 P3 或 holdout 执行，也没有 Git commit/push。
