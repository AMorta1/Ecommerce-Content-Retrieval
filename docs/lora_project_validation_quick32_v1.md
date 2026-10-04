# P7快速validation：固定分层32件人审

用户最新要求替代原200件全量人审安排；推理协议、600任务/模型、完整200件输出均不改变。选择32而非24，使每个二级品类固定4件，按源数据ID/品类、seed42、SHA256排序选择，不排除REVIEW/CONFLICT，不参考生成文案或人工结果。采样发生在Base生成已开始之后、任何人工质量结果之前，**不声称在全部生成之前**。

源数据：`data/processed/week1_v3/multimodal/validation.jsonl`（固定200件、8类各25）。实际规则与版本在 `configs/lora_project_validation_quick32_v1.json`；名单/hash在 `reports/generation/lora/project_validation_v1/quick32_v1/sample_ids.json` 与 `sampling_manifest.json`。不读取 project_test 或RAG holdout归档。32是本次LoRA人审样本，与RAG holdout32没有关系。

每类选4件后，按独立seed41008/namespace排序，2件Base为A、2件LoRA为A；总16/16。复核顺序用另一独立namespace，不按输出优劣调整。推理逐商品/任务seed仍沿用原600任务冻结值，不改采样后重新生成。

## 你现在需要做

1. 打开 `reports/generation/lora/project_validation_v1/quick32_v1/project_validation32_blind_review.xlsx`，读“复核说明”。
2. 在“盲评成对复核”看同一行源属性、原核心清单、实际输入和A/B三任务原文。灰色列不改、黄色列填写。需要调行高/展开完整文案列，可“审阅→撤销工作表保护”（无密码）。
3. A/B分别填 `matched_attribute_count`、`fluency_pass`、`factual_error_count`。命中以原核心清单为分母，全文任一正确表达即可，同字段一次；事实错误独立计点，跨任务同一错误一次；通顺独立判断。源争议写notes，不机械判幻觉或改分母。
4. 四个匿名偏好均选 `A更好/B更好/持平`：标题表达质量、卖点结构、短详情自然度、整体电商专业度。
5. 请特别填写两侧 `mechanical_template`（机械模板化）诊断0/1；其余四项诊断也建议填写：字段堆叠、无依据评价/效果/场景、数值范围失真、身份品类错误。诊断不替代正式指标；僵硬重复才是机械模板化，固定结构本身不自动算。错误或诊断1须notes引用原文及依据。
6. 填reviewer、review_confirmed=1，分批保存即可。全部32行完成后告诉我，再核验、先匿名汇总、后解盲。评审前勿查看匿名映射、版本输出或速度报告。

若由助手辅助初审，必须标记AI初审，不冒充人工最终标签；只有用户确认后才形成正式人工指标。本轮不自动执行任何test。

## 汇总与结论边界

正式人工分母仅这32件对应的原核心属性总数，报告命中率、错误样本率、平均独立事实错误数、通顺率、A/B偏好、模板化诊断，以及各品类4件结果；不把32件标签当200件标签。200件自动字面覆盖、结构、评价/数值疑似词、token上限与延迟可统计，但只是诊断，不替代人审。

“无明显事实性/属性覆盖退化且表达有收益”是用户最终判定，不自行添加新数值硬门槛。32件是快速方向性证据，不能宣称没有统计显著差异就等价或已解决幻觉。先汇报人工结果及源质量限制；用户确认候选后才可能进入project_test。当前adapter本身已经冻结为epoch3，本轮不改参数。

## 实际入口

在 `E:\Projects\Baidu\Ecommerce-Content-Retrieval`：

```powershell
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation_quick32.py --mode freeze
# 原Base/LoRA全量生成完成后：
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation_quick32.py --mode workbook
# 全32件人审填写并经确认后才运行：
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation_quick32.py --mode summarize
```

不运行原200件人审工作簿/汇总入口；原推理冻结文件不改。全量生成仍用 `scripts/run_lora_project_validation.py --mode generate --variant Base/LoRA`，无重生成。数据划分名称继续严格区分 lora_train、lora_train_dev、project_validation、project_test。
