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

| 人工指标 | 冻结LoRA | LoRA + RAG |
|---|---:|---:|
| core_attribute_hit_rate | 0.7472 | 0.7376 |
| fluency_pass_rate | 0.99 | 0.97 |
| factual_error_sample_rate | 0.02 | 0.01 |
| average_factual_errors_per_sample | 0.05 | 0.01 |

匿名偏好：
{
  "title_quality": {
    "LoRA_control": 5,
    "LoRA_RAG": 7,
    "tie": 88
  },
  "selling_points_structure": {
    "LoRA_control": 5,
    "LoRA_RAG": 0,
    "tie": 95
  },
  "short_description_naturalness": {
    "LoRA_control": 1,
    "LoRA_RAG": 0,
    "tie": 99
  },
  "overall_ecommerce_professionalism": {
    "LoRA_control": 17,
    "LoRA_RAG": 13,
    "tie": 70
  }
}

分品类人审：
{
  "LoRA_control": {
    "保温杯": {
      "sample_count": 13,
      "core_attribute_total": 101,
      "matched_attribute_total": 65,
      "core_attribute_hit_rate": 0.6435643564356436,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "垃圾桶": {
      "sample_count": 12,
      "core_attribute_total": 69,
      "matched_attribute_total": 60,
      "core_attribute_hit_rate": 0.8695652173913043,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "拖把": {
      "sample_count": 12,
      "core_attribute_total": 62,
      "matched_attribute_total": 54,
      "core_attribute_hit_rate": 0.8709677419354839,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.08333333333333333,
      "factual_error_total": 1,
      "average_factual_errors_per_sample": 0.08333333333333333
    },
    "收纳箱": {
      "sample_count": 13,
      "core_attribute_total": 81,
      "matched_attribute_total": 57,
      "core_attribute_hit_rate": 0.7037037037037037,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "移动电源": {
      "sample_count": 12,
      "core_attribute_total": 77,
      "matched_attribute_total": 54,
      "core_attribute_hit_rate": 0.7012987012987013,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.08333333333333333,
      "factual_error_total": 4,
      "average_factual_errors_per_sample": 0.3333333333333333
    },
    "耳机": {
      "sample_count": 13,
      "core_attribute_total": 76,
      "matched_attribute_total": 62,
      "core_attribute_hit_rate": 0.8157894736842105,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "键盘": {
      "sample_count": 13,
      "core_attribute_total": 85,
      "matched_attribute_total": 58,
      "core_attribute_hit_rate": 0.6823529411764706,
      "fluency_pass_rate": 0.9230769230769231,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "鼠标": {
      "sample_count": 12,
      "core_attribute_total": 74,
      "matched_attribute_total": 57,
      "core_attribute_hit_rate": 0.7702702702702703,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    }
  },
  "LoRA_RAG": {
    "保温杯": {
      "sample_count": 13,
      "core_attribute_total": 101,
      "matched_attribute_total": 65,
      "core_attribute_hit_rate": 0.6435643564356436,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "垃圾桶": {
      "sample_count": 12,
      "core_attribute_total": 69,
      "matched_attribute_total": 54,
      "core_attribute_hit_rate": 0.782608695652174,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "拖把": {
      "sample_count": 12,
      "core_attribute_total": 62,
      "matched_attribute_total": 54,
      "core_attribute_hit_rate": 0.8709677419354839,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "收纳箱": {
      "sample_count": 13,
      "core_attribute_total": 81,
      "matched_attribute_total": 56,
      "core_attribute_hit_rate": 0.691358024691358,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "移动电源": {
      "sample_count": 12,
      "core_attribute_total": 77,
      "matched_attribute_total": 53,
      "core_attribute_hit_rate": 0.6883116883116883,
      "fluency_pass_rate": 0.9166666666666666,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "耳机": {
      "sample_count": 13,
      "core_attribute_total": 76,
      "matched_attribute_total": 59,
      "core_attribute_hit_rate": 0.7763157894736842,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    },
    "键盘": {
      "sample_count": 13,
      "core_attribute_total": 85,
      "matched_attribute_total": 64,
      "core_attribute_hit_rate": 0.7529411764705882,
      "fluency_pass_rate": 0.8461538461538461,
      "factual_error_sample_rate": 0.07692307692307693,
      "factual_error_total": 1,
      "average_factual_errors_per_sample": 0.07692307692307693
    },
    "鼠标": {
      "sample_count": 12,
      "core_attribute_total": 74,
      "matched_attribute_total": 56,
      "core_attribute_hit_rate": 0.7567567567567568,
      "fluency_pass_rate": 1.0,
      "factual_error_sample_rate": 0.0,
      "factual_error_total": 0,
      "average_factual_errors_per_sample": 0.0
    }
  }
}

重点事实错误/诊断记录见human_diagnostic_cases.json；不据case调整或重跑。

## 冻结与停止

生成前统计：{"products": 100, "by_category": {"移动电源": 12, "耳机": 13, "键盘": 13, "鼠标": 12, "保温杯": 13, "垃圾桶": 12, "拖把": 12, "收纳箱": 13}, "original_core_denominator": 625, "source_quality": {"REVIEW": 11, "PASS": 84, "CONFLICT": 5}, "original_input_core_fields": 567, "mandatory_core_facts_per_product_sum": 350, "identity_core_facts_sum": 162, "max_input_tokens": 832}
协议/代码/数据/Prompt/adapter/环境hash见protocol_manifest.json，原始代码字节也已独立归档。
原RAG v2 P3未进入；32条holdout清单未读取、未生成，原归档不变。没有训练、调参、commit或push。
本轮无论结果好坏只归档，禁止修改后重跑同批test。
