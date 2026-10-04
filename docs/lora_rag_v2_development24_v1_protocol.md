# 冻结 LoRA v1 + RAG v2 事实机制：独立组合验证

用户批准的独立组合实验，不重训，不进入原 RAG v2 P3，不创建 P2.7/P2.8，不修改历史结果。仅沿用已知非 holdout 的 development24（8个二级品类各3件），不是新盲测，也不是正式 test。禁止读取 project_test、32条 RAG v2 holdout 清单或运行该 holdout。

## 对照与组合

- LoRA_control：复用原 P7 同24件的 LoRA 三任务输出，只在全部源数据、Prompt、seed、preflight、adapter、环境及输出 hash 验证后使用。不给历史人工标签背书，不导入旧人工评分。
- LoRA_RAG：相同 epoch 3 adapter（checkpoint-step-001359）、固定本地 Qwen revision、NF4/double quant/BF16、三任务、max tokens、解码与逐商品/任务 seed。完整保留 P7 可靠核心属性，增加冻结 policy v4 选择的四区事实。原始标题不进入输入。
- 不把 Top-3 当全部输入，不绕过 REVIEW/CONFLICT/placeholder 门控，不恢复 P7 已排除的核心字段。P7 与 RAG v2 既有多值门控不同：原 P7 输入保持原样；该差异单独记录，不能虚报全部原核心输入都经过 RAG v2 门控。
- 必要接口适配仅将系统约束“可靠核心属性”改为“可靠属性”，以允许非核心 Supplemental；P5 三任务要求不变，追加中性四区标签。不继承旧整份 JSON Prompt、V2-coverage 强覆盖或 P2.6 语言约束。标题仍沿用 P5 的核心属性限定。
- 本实验测量“追加事实及其四区组织/必要接口适配”的整体增量，不声称单独识别检索算法的因果贡献，也不与旧一次调用 RAG 分数/耗时混比。

## 冻结、检查与一次运行

在第一次新生成前保存24件ID、8件工程样本ID、匿名映射、所有真实三任务 messages/seed/token hash、input/context/config/code/adapter/model metadata/chat-template/environment hash。工作区未提交时保留独立代码原字节快照和 Git HEAD/status。只离线加载固定 snapshot；不安装依赖。

tokenizer 禁用截断，检查全部原核心 JSON 及每条检索事实（包括 Mandatory 与 identity coverage）实际完整进入模型输入；过长、缺失、重复、跨商品或不可靠新事实直接报错。Mandatory 注入分母只含源实际存在且 eligible 的 Mandatory facts；missing 不算失败，不改原 core_attribute_count。零分母率为 null。

8件/24调用 smoke 仅检查运行、token、解析状态和日志，不阅读质量再调内容。若工程异常立即停止，不自动重试。smoke 通过后仅生成一次 development24/72调用；原对照不重新生成。smoke 结果单独保存，不计入正式24件指标。

## 评测与停止

正式人工三指标沿用 validation：matched_attribute_count / fluency_pass / factual_error_count，以整份三任务串联文案为单位，原核心字段为分母，每字段最多一次，同义/格式表达可命中；独立事实错误跨任务重复只计一次。事实错误不自动判不通顺，源质量问题单列，自动 Grounding 只辅助。

匿名A/B逐商品比较标题质量、卖点结构、短详情自然度、整体电商专业度；记录字段堆叠、模板化、无依据扩写、数值范围失真、身份品类错误。源属性、原核心清单、两种实际输入、四区事实与A/B三任务在同一行；填写区黄色，所有人工值初始留空。隐藏匿名映射不等于加密，复核前不查看。

自动统计解析、字面覆盖、疑似评价/效果/场景/数值、三任务与总耗时、RAG选择耗时、token与峰值显存；使用冻结旧 validator，绝不删除、改写或自动重生。原对照的 latency 是历史测量，不与新硬件负载做严格因果归因。RAG选择耗时单独记录；预计算/预检时间及模型加载不混入模型生成耗时。

人工复核完成前 human metrics/prefs 为 null，不以自动指标宣告组合成功。全部确认、匿名汇总保存后才解盲，给出总体和各品类结果。小样本不做总体显著提升宣称。不根据结果再调 Prompt/LoRA，不运行 test；完成自动生成与工作簿后先停止等待复核。

## 命令（本机）

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_rag_integration.py --mode prepare
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_rag_integration.py --mode smoke
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_rag_integration.py --mode generate
& 'E:\Anaconda\python.exe' -B -X utf8 scripts/run_lora_rag_integration.py --mode review
```

最后一条只构造初始空白人审工作簿。人工复核保存后，另行确认才执行 `--mode summarize`；该命令拒绝不完整评分、篡改固定源/输出、重复商品或提前解盲。不自行 commit/push。
