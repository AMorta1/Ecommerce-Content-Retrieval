# LoRA v1 test100 正式生成交付与验收

本轮已经完成一次性正式生成和自动归档；**最终人工评测尚未完成**。状态为 `formal_test100_generation_complete_human_review_pending`，不能将自动指标当作人工三指标或给出最终偏好结论。

## 已完成

- 原冻结test100全部100件，每模型300任务，总600次调用；独立输出和组装文案均保存。
- 正式入口独立于validation，仅复用冻结的通用函数。34项合成单元测试通过，其中对模型加载至生成结束校验的执行体作AST比对，确认与P7一致，仅继承配置变量名不同。
- 300组seed、Prompt hash、preflight全部一致；推理配置差异仅variant和adapter。
- best仍为epoch3/checkpoint-step-001359，adapter权重SHA-256为 `890a061812321ccf6bbf598b37617fc59dcde70d57042c4304c219a0f6740e31`。模型参数未更新。
- 输入门控与原validation200件逐条核对等价。全100件保持原核心分母625，567个字段进入可靠输入；源审计84 PASS/11 REVIEW/5 CONFLICT，不剔除商品或缩小评测分母。
- Base运行512.646秒，LoRA运行514.467秒左右（精确值见各自run_report.json）；均无工程异常、重试、续跑或网络加载。
- 34个冻结文件、22个自动阶段归档文件校验通过，工作簿固定列与冻结源/输出逐单元格一致；100对的人审与偏好列留空，IDs为文本，映射保持veryHidden。
- 工作簿生成用已有 `E:\Anaconda\python.exe`（有openpyxl），推理用原 `ecommerce-generation` 环境。没有安装依赖或升级环境；实际命令与原因在execution_plan.json，纠正了冻结协议文档中archive/summarize的解释器示例，未改变任何推理配置。

## 自动对照（仅辅助）

| 自动项 | Base | LoRA |
| --- | ---: | ---: |
| 标题解析 | 100/100 | 100/100 |
| 卖点解析 | 46/100 | 99/100 |
| 短详情解析 | 100/100 | 100/100 |
| 三任务组装结构 | 46/100 | 99/100 |
| 字面核心覆盖 | 519/625（83.04%） | 493/625（78.88%） |
| 标题平均秒 | 1.163 | 1.190 |
| 卖点平均秒 | 1.850 | 1.594 |
| 短详情平均秒 | 1.886 | 2.211 |
| 三任务总平均秒 | 4.899 | 4.994 |
| 空输出/触及max token | 0/0 | 0/0 |
| 疑似评价词样本 | 80 | 4 |
| 疑似数值样本 | 1 | 1 |

完整分品类自动统计、P50/P95、效果/场景疑似词、显存见automatic_phase_summary.json及automatic_phase_report.md。词汇可能有源支持，疑似数值可能是正常格式；这些不是事实错误计数。耗时含冷启动且输出长度可能不同，不单独归因于adapter，也不与旧单调用Baseline直接归因比较。

## 范围诊断限制及人工复核重点

冻结的补充范围线索仅匹配 `[6,12)` 等数字区间写法，未识别实际源字段中的“6小时(含)-12小时(不含)”等中文区间。`range_boundary_clues_REVIEW_ONLY_samples=0` **不可用来推断无范围错误，也不是有效的中文区间完整审计**。保留原自动结果，不覆盖、修补或重跑生成，也不修改冻结的检测代码。

`range_review_focus_blind.csv` 只依据源核心属性中的“含/不含”标记列出匿名配对ID、字段和原始值，不检查哪一侧更好、不填写判断。它不改变全100件人工范围。复核时要同时查看A/B全文，判断是否扩大/缩小范围或丢失边界；不能以字面差异直接判错，正常等义表达仍允许。源属性互相冲突时在notes单列，不机械算生成幻觉。

## 你现在需要做

1. 打开 `project_test100_blind_review.xlsx`，先看“复核说明”。评审结束前不要查看匿名映射或版本标识输出/报告。
2. 对全部100行，A/B分别填写 matched_attribute_count、fluency_pass、factual_error_count；核心分母不改。
3. 四项偏好填写A更好/B更好/持平；诊断与具体错误原文/依据写notes。字段标签、固定中性结构本身不自动算机械模板化。
4. 填reviewer、review_confirmed=1，保存，然后通知复核完成。若助手辅助初审，应保留AI来源标记，只有用户最终确认后才归为最终人工结果。
5. 全100对最终确认后只运行下面汇总入口，不再运行prepare/generate：

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode summarize
```

人审完成后形成blind_summary.json（先匿名汇总）、human_metrics_unblinded.json、human_review_final.csv、human_bad_cases.json、final_test_report.md、final_test_manifest.json。当前这些正式人工产物尚不存在，最终三指标、匿名偏好、分品类人工结果及重点确认bad case均待复核，不能编造。

## 边界

这是原project_test中的固定100件评测，不是200件全量。抽样和ID沿用历史冻结清单，没有重新挑选；样本此前被Baseline/RAG评估，本轮不称首次完全未观察测试。P5数据、正式训练config、adapter、Prompt、chat template、decoding及seed不改。没有读取RAG v2 holdout归档或运行RAG v2；没有commit/push。不再回到validation调参，不依据本次test调整模型或Prompt。完成当前交付后停止，等待人审。
