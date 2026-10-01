# formal RAG v1 test100 二次规则复核摘要

## 范围

- 仅复核人工标注草稿；未运行模型，未修改formal输出、冻结配置、Prompt、Top-K或原始初标CSV。
- Validator只作为线索；事实判断同时参考完整结构化属性、原标题、RAG上下文和既有只读质量审计。
- 本文件不计算最终正式指标，二次结果仍等待人工最终确认。

## 输入完整性

- `reports/generation/rag/formal_rag_v1_test100_human_review.csv`: `58c37006e7720cc8fef6ab8806281136695587c88722d264a66673bfb9551eb6`
- `reports/generation/rag/formal_rag_v1_test100_outputs.json`: `36edf46d4a4db42254713a823265a36937439860e3c3593dad86a37976bbc298`
- `reports/generation/rag/formal_rag_v1_test100_validator.json`: `2c83d36d5da7b3bd7fd0f6e96e3c405e53e9169a94f237ee5930a9be92339d36`
- `data/processed/week2_rag_formal_v1/test100_fact_units.jsonl`: `5deb2934bbe9500235c034a700d89ba0c5ea6920dd2c37cb3eaeeb051a94b3b8`
- `reports/data/week2_rag_fact_quality_audit.csv`: `887ca944c9725f92b213687f1fbebbed82effd8f2bf3f3ae8734eb3e8e290ce0`

## 修改统计

- 原初标行数：100
- 至少一个评分/归因字段发生修改的商品数：77
- 三个优先字段发生修改的商品数：51
- 逐字段变更记录数：174
- 删除机械Top-3漏召回计数的商品数：47
- 高优先级待人工确认商品数：19
- Grounding/Unsupported归因暂留空的商品数：4

字段变更数：

- `matched_attribute_count`: 29
- `fluency_pass`: 7
- `factual_error_count`: 33
- `category_style_pass`: 3
- `source_fact_quality_count`: 3
- `retrieval_error_count`: 47
- `grounding_failure_count`: 5
- `unsupported_generation_count`: 33
- `supported_paraphrase_count`: 14

## 主要系统性修正

1. 将‘未进入Top-3的核心属性数’与Retrieval Error解耦；没有明确选错证据的行统一为0。
2. 同一事实外扩在卖点和详情机械重复时按一个独立错误点计数。
3. 使用完整可靠商品事实复核，清除被原标题/完整属性支持的Validator误报。
4. 补标Validator漏掉的身份错位、范围事实收窄、无来源评价/效果/场景和新增参数。
5. 将结构成功与流畅性分开，补标异常重复、字段堆叠和‘光机电光电’等问题。
6. 对品类直接语义释义使用一致口径；家居收纳、日常清洁等安全释义不计幻觉。

## 待人工确认

以下商品包含既有REVIEW/CONFLICT，或本轮发现的明确源事实质量线索：

`2859211238`, `522783084872`, `540395966499`, `544826941132`, `545593037598`, `566531828259`, `583014299804`, `583457250405`, `589508217209`, `591309078741`, `599485472276`, `600661711292`, `608516760684`, `609629279071`, `609762128375`, `613735876059`, `621173570641`, `621846828786`, `624392441174`

其中归因字段暂留空（不硬分Grounding/Unsupported）的商品：

`540395966499`, `589508217209`, `608516760684`, `613735876059`

## 输出

- 修订草稿：`reports/generation/rag/formal_rag_v1_test100_secondary_review.csv`
- 逐字段变更日志：`reports/generation/rag/formal_rag_v1_test100_secondary_review_changes.csv`
