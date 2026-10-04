# LoRA v1 正式 test100 自动阶段报告

状态：完整100件配对生成已完成，最终人工复核待完成。不是全200件test评测。

测试样本此前用于Baseline/RAG，不宣称首次完全未观察测试。本轮Base与LoRA均为同配置三任务推理，不与旧单调用Baseline直接比较。

| 自动辅助项 | Base | LoRA |
| --- | ---: | ---: |
| 商品数 | 100 | 100 |
| 任务调用 | 300 | 300 |
| 组装结构成功 | 46 | 99 |
| 空输出任务 | 0 | 0 |
| max token任务 | 0 | 0 |
| 疑似评价词商品 | 80 | 4 |
| 疑似效果/范围词商品 | 84 | 3 |
| 场景词商品（可能有来源） | 47 | 15 |
| 疑似数值商品 | 1 | 1 |
| 区间边界字面缺失线索商品 | 0 | 0 |

## Base 自动诊断

三任务解析成功：{'title': 100, 'selling_points': 46, 'short_description': 100}。
字面覆盖：519/625（83.04%），不是人工属性命中率。
峰值allocated/reserved：5497.7/5628.0MiB。

| 任务 | 平均秒 | P95秒 |
| --- | ---: | ---: |
| title | 1.163 | 1.860 |
| selling_points | 1.850 | 2.939 |
| short_description | 1.886 | 3.191 |
| three_task_total | 4.899 | 7.964 |

## LoRA 自动诊断

三任务解析成功：{'title': 100, 'selling_points': 99, 'short_description': 100}。
字面覆盖：493/625（78.88%），不是人工属性命中率。
峰值allocated/reserved：5507.4/5636.0MiB。

| 任务 | 平均秒 | P95秒 |
| --- | ---: | ---: |
| title | 1.190 | 2.031 |
| selling_points | 1.594 | 2.461 |
| short_description | 2.211 | 3.295 |
| three_task_total | 4.994 | 7.875 |

## 下一步人工操作与结论边界

打开 `project_test100_blind_review.xlsx`，对全100行分别填写两侧三正式指标、四匿名偏好、诊断及依据，reviewer、review_confirmed=1。
评审完成前不要查看veryHidden映射或按版本命名的输出/报告。人工字段全部留空，自动线索不能代替人审。
分品类自动结果见automatic_phase_summary.json；正式人工三指标、匿名偏好、分品类人工结果及重点bad case均待最终复核，当前不编造。
未根据输出调参、重试、更新权重；不运行RAG v2或读取其holdout。保留原始解析失败输出。
本轮自动完成后停止，等用户复核；后续仅汇总归档，不因test结果修改模型/Prompt。
