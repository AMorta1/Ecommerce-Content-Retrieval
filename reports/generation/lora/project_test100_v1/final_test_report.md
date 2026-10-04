# LoRA v1 最终test100人工报告

完整100对最终确认，指标与validation口径一致。

既有test100此前用于Baseline/RAG，本轮不是首次未观察测试；不根据结果调参或重跑。

| 人工指标 | Base | LoRA |
| --- | ---: | ---: |
| core_attribute_hit_rate | 0.8896 | 0.7472 |
| fluency_pass_rate | 0.98 | 0.99 |
| factual_error_sample_rate | 0.92 | 0.01 |
| factual_error_total | 372 | 5 |
| average_factual_errors_per_sample | 3.72 | 0.05 |

## 匿名偏好解盲

```json
{
  "title_quality": {
    "Base": 93,
    "LoRA": 6,
    "tie": 1
  },
  "selling_points_structure": {
    "Base": 7,
    "LoRA": 93,
    "tie": 0
  },
  "short_description_naturalness": {
    "Base": 98,
    "LoRA": 2,
    "tie": 0
  },
  "overall_ecommerce_professionalism": {
    "Base": 8,
    "LoRA": 91,
    "tie": 1
  }
}
```

分品类和诊断见human_metrics_unblinded.json，人工确认的重点案例见human_bad_cases.json（排序只用于报告，不用于挑评测样本）。
自动指标、每任务和总延迟见automatic_phase_summary.json。人工标签不由自动线索替代。

至此只归档，不根据test修改模型、Prompt、训练数据或推理参数。
