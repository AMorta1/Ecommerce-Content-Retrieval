# LoRA + RAG 追加固定 test100 正式对照

原冻结test100此前已用于Baseline、RAG及Base/LoRA评测，并已被观察。本次为用户批准的追加固定组合版本正式对照，不是新的未观察盲测；A/B仅表示版本匿名。组合内容在读取本轮test输出前冻结，无论结果好坏只归档，不修改后重跑。

## 状态与可比性

同原100件、同625核心字段分母。control复用已冻结LoRA三任务输出，组合仅生成一次300任务。
两臂同adapter/基座、task、max tokens、解码与逐商品逐任务seed；处理差异仅为已冻结的四区上下文及必要system接口适配。
历史control耗时非同期测量，不直接作纯检索因果归因；自动线索不是事实错误率或通顺率。

| 自动辅助指标 | 冻结LoRA | LoRA + RAG |
|---|---:|---:|
| 组装结构成功/100 | 99 | 99 |
| 空任务/300 | 0 | 0 |
| max-token任务/300 | 0 | 0 |
| 疑似评价样本（非事实错误） | 4 | 3 |
| 疑似数值样本（非事实错误） | 1 | 0 |
| 峰值allocated MiB | 5507.4 | 5692.1 |
| title 解析成功/100 | 100 | 100 |
| selling_points 解析成功/100 | 99 | 99 |
| short_description 解析成功/100 | 100 | 100 |

### LoRA_control

字面覆盖（非人审）：{'matched': 493, 'denominator': 625, 'rate': 0.7888}
各任务和三任务总耗时：{"title": {"mean": 1.1901300630001423, "p95": 2.031343654999546, "max": 2.2169994000032602}, "selling_points": {"mean": 1.5936684820001028, "p95": 2.4613087950003316, "max": 2.794176299998071}, "short_description": {"mean": 2.2106843890001984, "p95": 3.2952246400005603, "max": 4.060579900000448}, "three_task_total": {"mean": 4.994482934000444, "p95": 7.8753187799980875, "max": 8.59382469999764}}
输入tokens：mean=126.26, max=177
Mandatory实际token注入：{'expected': 0, 'verified': 0, 'rate': None}
效果/范围疑似样本=3；场景疑似样本=15；范围边界线索样本=0
分品类自动辅助：{"保温杯": {"products": 13, "literal_matched": 65, "core_denominator": 101}, "垃圾桶": {"products": 12, "literal_matched": 60, "core_denominator": 69}, "拖把": {"products": 12, "literal_matched": 54, "core_denominator": 62}, "收纳箱": {"products": 13, "literal_matched": 57, "core_denominator": 81}, "移动电源": {"products": 12, "literal_matched": 54, "core_denominator": 77}, "耳机": {"products": 13, "literal_matched": 63, "core_denominator": 76}, "键盘": {"products": 13, "literal_matched": 83, "core_denominator": 85}, "鼠标": {"products": 12, "literal_matched": 57, "core_denominator": 74}}

### LoRA_RAG

字面覆盖（非人审）：{'matched': 472, 'denominator': 625, 'rate': 0.7552}
各任务和三任务总耗时：{"title": {"mean": 1.432746119999647, "p95": 2.123562999998466, "max": 2.4678664000020945}, "selling_points": {"mean": 1.881394071000177, "p95": 2.8667770799977004, "max": 3.729734599997755}, "short_description": {"mean": 2.239510693000193, "p95": 3.5708351900033444, "max": 4.274791299998469}, "three_task_total": {"mean": 5.553650884000017, "p95": 8.773445459999493, "max": 9.950133699996513}, "three_task_plus_retrieval": {"mean": 5.55385770300054, "p95": 8.773612589997356, "max": 9.950307799997972}, "three_task_plus_retrieval_and_preflight": {"mean": 5.561951086000408, "p95": 8.782881125006316, "max": 9.960825300004217}}
输入tokens：mean=544.64, max=832
Mandatory实际token注入：{'expected': 1050, 'verified': 1050, 'rate': 1.0}
效果/范围疑似样本=5；场景疑似样本=11；范围边界线索样本=0
分品类自动辅助：{"保温杯": {"products": 13, "literal_matched": 65, "core_denominator": 101}, "垃圾桶": {"products": 12, "literal_matched": 54, "core_denominator": 69}, "拖把": {"products": 12, "literal_matched": 54, "core_denominator": 62}, "收纳箱": {"products": 13, "literal_matched": 56, "core_denominator": 81}, "移动电源": {"products": 12, "literal_matched": 53, "core_denominator": 77}, "耳机": {"products": 13, "literal_matched": 60, "core_denominator": 76}, "键盘": {"products": 13, "literal_matched": 74, "core_denominator": 85}, "鼠标": {"products": 12, "literal_matched": 56, "core_denominator": 74}}

## 正式人工指标

待完成100行A/B人工复核；目前不能给出组合正式核心命中率、通顺率、事实错误样本率或偏好结论。
填写lora_rag_project_test100_blind_review.xlsx黄色列：两臂各三指标、四项偏好、reviewer、review_confirmed=1。
旧LoRA正式标签不预填到本轮工作簿，避免破坏配对匿名评审；既有结果保持原样。

## 冻结与停止

生成前统计：{"products": 100, "by_category": {"移动电源": 12, "耳机": 13, "键盘": 13, "鼠标": 12, "保温杯": 13, "垃圾桶": 12, "拖把": 12, "收纳箱": 13}, "original_core_denominator": 625, "source_quality": {"REVIEW": 11, "PASS": 84, "CONFLICT": 5}, "original_input_core_fields": 567, "mandatory_core_facts_per_product_sum": 350, "identity_core_facts_sum": 162, "max_input_tokens": 832}
协议/代码/数据/Prompt/adapter/环境hash见protocol_manifest.json，原始代码字节也已独立归档。
原RAG v2 P3未进入；32条holdout清单未读取、未生成，原归档不变。没有训练、调参、commit或push。
本轮无论结果好坏只归档，禁止修改后重跑同批test。
