# LoRA + RAG 独立 development24 验证

自动生成完成，人工复核待完成。**不以自动 Grounding 判定事实改善。**

同24件、原核心分母134；每版本三任务。对照复用已冻结P7输出，组合新增72调用，smoke另24调用不计分。

| 自动辅助指标 | LoRA control | LoRA + RAG |
|---|---:|---:|
| 三任务组装结构成功 | 24 | 24 |
| 疑似评价/效果样本（非错误率） | 0 | 0 |
| Grounding PASS（非人审） | 24 | 24 |
| 峰值显存 MiB（allocated） | 5503.6 | 5642.8 |

## LoRA_control

字面覆盖（非人工）：{'matched': 116, 'denominator': 134, 'rate': 0.8656716417910447}
三任务解析成功数：{'title': 24, 'selling_points': 24, 'short_description': 24}
三任务总耗时：{'mean': 5.187997341666384, 'p95': 8.15048087999603, 'max': 8.329211300002498}
Mandatory token 注入：{'expected': 0, 'verified': 0, 'rate': None}

## LoRA_RAG

字面覆盖（非人工）：{'matched': 112, 'denominator': 134, 'rate': 0.835820895522388}
三任务解析成功数：{'title': 24, 'selling_points': 24, 'short_description': 24}
三任务总耗时：{'mean': 5.791524929166068, 'p95': 8.778039410005658, 'max': 9.967619700000796}
Mandatory token 注入：{'expected': 228, 'verified': 228, 'rate': 1.0}

## 边界与下一步

人工 matched_attribute_count / fluency_pass / factual_error_count 和四项匿名偏好均未预填，当前不能判断组合是否更优。
人工工作簿：lora_rag_development24_blind_review.xlsx。只填写黄色列，完整确认后再汇总。
旧对照 latency 是历史测量；组合增量包含中性四区事实组织及系统约束从可靠核心属性到可靠属性的必要适配，不是纯检索因果隔离。
P7既有耳机类别多值输入与RAG门控存在1项差异，已记录protocol_manifest；原输入/分母不改，也不将该字段重复注入RAG区。
无训练、无test读取、无holdout清单读取/生成；原RAG v2归档与frozen_not_executed状态不改。没有commit/push。
