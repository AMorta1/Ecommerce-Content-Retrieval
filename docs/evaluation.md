# 检索人工评测说明

## 当前状态

当前评测版本是 `week1_v3_validation_pool_v1`：

- 从 validation 的 200 件商品中检索。
- 24 条中文查询，覆盖 8 个二级品类，每类 3 条。
- 每条查询保留前 20 个候选，共 480 行。
- 480 条候选已经全部完成人工复核，当前不存在空标签或 AI 初标标记。

正式基线已经生成：Precision@10 为 49.58%，pooled Recall@10 为 67.08%，MRR@10 为 73.70%，NDCG@10 为 63.53%。这个候选池用于 Week 2 开发和优化前后对比；方案基本确定后，Week 3 再建立独立 test 评测。

## 需要填写的文件

用 Excel 打开：

~~~text
reports/retrieval/evaluation/annotation_pool.csv
~~~

人工复核时只修改最后两列：

- `relevance_grade`：必填，只能是 0、1 或 2。
- `review_notes`：可选，用于记录用途、材质、规格冲突或修改原因。

不要修改 `query_id`、`candidate_rank`、`product_id` 等结构字段。`title` 是标题，`candidate_attributes` 只展示与当前查询有关的来源属性，`image_path` 是本地商品图片。

`product_id` 是长数字字符串，Excel 保存 CSV 时可能将它变成科学计数法并改坏末尾数字。每次保存后可运行：

~~~powershell
python scripts/repair_csv.py reports/retrieval/evaluation/annotation_pool.csv
~~~

如果脚本报告错误，再根据同一行图片文件名恢复：

~~~powershell
python scripts/repair_csv.py --repair reports/retrieval/evaluation/annotation_pool.csv
~~~

修复不会清空人工标签。评测代码也会拦截商品编号和图片文件名不一致的表。

## 0、1、2 如何判断

| 标签 | 含义 | 判断方法 |
| --- | --- | --- |
| 2 | 高度相关 | 品类正确，并满足查询的主要属性和用途 |
| 1 | 部分相关 | 核心品类正确，只满足部分要求，且没有明确违背最关键条件 |
| 0 | 不相关 | 品类错误，或与关键要求明确冲突 |

例如查询“USB 接口有线机械键盘”：

- USB、有线、机械键盘：2。
- 是有线键盘，但无法确认是否机械键盘：1。
- 无线或蓝牙键盘，与“有线”冲突：0。
- 鼠标、耳机等错误品类：0。

如果标题、属性和图片互相矛盾，不要自行猜测。只是不确定可填 1 并写明冲突；关键条件明确相反则填 0。

## AI 辅助初标的保护机制

如果将来生成新的 AI 辅助初标，程序会通过 `review_notes` 中的“AI初标”标记阻止正式评测。只查看临时趋势时必须显式增加参数：

~~~powershell
python scripts/evaluate_retrieval.py --allow-assistant-draft
~~~

生成的报告会写明 AI 初标数量，只能用于初步分析，不能作为 PRD 达标结论。当前版本已经完成人工复核，不需要使用这个参数。

## 完成人工复核后计算正式指标

~~~powershell
python scripts/evaluate_retrieval.py
~~~

默认命令会拒绝空标签和“AI初标”标记。当前正式结果生成于：

~~~text
reports/retrieval/evaluation/baseline_metrics.json
~~~

指标包括：

- Precision@10：前 10 个结果中标签为 1 或 2 的比例。
- pooled Recall@10：前 10 个结果覆盖了候选池中多少相关商品。
- MRR@10：第一个相关结果出现得有多靠前。
- NDCG@10：同时考虑排名位置和 0/1/2 相关程度。

## 为什么只能叫 pooled Recall

目前只人工判断每条查询的前 20 个候选，没有判断 validation 全部 200 件商品。因此不知道完整商品库里究竟有多少相关商品，Recall 的分母只能使用这 20 个候选中被判相关的商品数。

加入类别重排或新模型后，应把新方案 Top-20 中未出现过的商品追加到共同候选池，再补充人工标签。所有方案在同一个扩充候选池上重新计算，才是公平对比。不能把候选池 Recall 包装成覆盖全库的正式 Recall。

## 重新生成候选池

当前表已经包含人工填写内容，正常情况下不要重新生成。脚本检测到已有表时会拒绝覆盖。只有创建了新的评测版本和新输出路径后，才运行：

~~~powershell
python scripts/prepare_retrieval_evaluation.py
~~~

评测版本、查询文件、候选深度和输出位置统一配置在 `configs/retrieval_evaluation.json`。

## 质量要求

- 正式汇报前，本人应快速复核 24 条查询是否像真实购物表达。
- 最理想是两人独立标注并检查一致性；四周条件下，至少随机复核 10% 标签并记录修改数量。
- 不允许按“候选品类相同”自动打标签，有线/无线、容量、材质等细粒度条件同样重要。
- validation 用于开发和选方案；最终 test 查询和标签必须等方案基本确定后再建立。

## 跨模块实验记录

生成和检索的正式评测结果统一登记到 `reports/experiment_log.csv`，入口为 `scripts/track_experiment.py`。脚本直接读取指标文件，不手工复制数值；同时将配置内容保存到 `parameters_json`，并记录指标、配置清单的路径和哈希。

实验表中的差值只在模块和 `evaluation_set_id` 都相同时计算。生成评测集由固定100条样本的哈希识别；检索评测集由查询文件哈希、候选库划分和候选库数量共同识别。若评测集不同，`comparison_status` 会写为 `no_matching_baseline`，各差值列保持空白。

差值列默认是“当前版本减 Baseline”；只有 `factual_error_rate_reduction` 使用“Baseline减当前版本”，因此该列为正数表示事实错误率下降。`delta_average_latency_seconds` 为正数表示新版本更慢。
