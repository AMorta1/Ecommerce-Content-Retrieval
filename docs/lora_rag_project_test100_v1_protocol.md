# LoRA + RAG 追加固定 test100 对照协议

用户于2026-10-05批准本轮：沿用原100件，对照冻结LoRA三任务输出；冻结现有组合，不调整，结果无论好坏只归档。

## 历史与解释边界

这批project_test已用于Baseline/RAG、Base/LoRA并被观察。本轮不是新的未观察盲测，也不提供未见测试集的独立泛化证据。A/B是版本匿名，不代表数据从未被观察；展示实际输入事实后可能识别增强方式，因此不宣称完全双盲。

本轮是独立组合版本追加正式test100，不是原RAG v2 P3晋级；原RAG v2归档及32条generation-output holdout保持frozen_not_executed，不读取其清单，不运行其生成。测试100件与8品类范围不重抽，不根据输出选样本。

## 两个固定方案

- LoRA_control：复用此前hash验证的epoch3 checkpoint-step-001359正式test100输出，不重新生成、不导入旧人工标签。
- LoRA_RAG：同一固定基座、adapter、三任务、chat template、解码和逐商品/逐任务seed；复用development24冻结的四区接口、rag_fact_policy_v4、Top-3及与原输入门控取交集规则，不修改内容Prompt。

两臂唯一处理差异是四区上下文及既有必要接口适配：原system的“可靠核心属性”放宽为“可靠属性”，原任务说明/可靠核心JSON不变，追加中性四区标签。不是“除adapter外一切相同”的Base/LoRA实验，也不是纯检索算法因果隔离。历史control耗时与新组合耗时不作严格同期因果比较。

## test事实构造与禁止事项

事实仅来自原冻结generation_evaluation_samples.jsonl的完整结构化属性及既有test100源质量审计。仅在本轮独立输出目录构造test100_fact_units.jsonl；冻结策略在train+validation制定，test不参与策略修改。build_fact_units要求split一致，因此将既有审计scope=test100在内存副本映射为test；不改变源文件、状态、blocked_fields或备注。所有事实保留源hash与审计证据。

不把原生标题、旧模型输出、人工标签作为模型输入；不继承smoke adapter；不训练、不升级依赖、不调Prompt/解码、不修改正式评测分母。原可靠核心保留；RAG不得恢复原门控已撤回的核心字段。缺失源事实不作为Mandatory注入失败。原输入与RAG质量门控的差异完整记录，不静默“修复”。

既有test源审计有1件identity_block，development24未遇到该结构性分支。本轮保持全字段撤回，只让blocked_conflict事实作为selector元数据保留商品ID，四区实际注入全部为空；不放行事实、不改策略，原核心分母仍保留。此处只修复空上下文ID校验边界，不根据生成质量改Prompt。

## 生成前冻结与运行

先单元测试，再prepare：验证既有两条协议及源输出hash，冻结代码原始字节、Git HEAD/dirty状态、全部数据/config/adapter/policy、模型本地snapshot/revision、环境、chat template、全部300份真实Prompt及token preflight、ID顺序和匿名映射。

所有Mandatory/Identity/Supplemental/Negative事实以及原核心JSON必须在tokenizer后完整存在；不截断、不裁剪。溢出或缺失直接报错，不记录虚假的100%。无Mandatory时单商品比率为null。

已完成development24工程smoke，本轮不对test进行额外smoke生成。静态全量preflight完成后只消费一次正式300调用。输出目录存在即拒绝重新生成/resume；异常停止留痕，不自行修参数或重跑。固定本地snapshot、离线NF4/double-quant/BF16推理、q_proj/v_proj rank8 adapter，不更新参数。

```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_rag_project_test.py --mode prepare
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_rag_project_test.py --mode generate --confirm-fixed-test100
& 'E:\Anaconda\python.exe' -B -X utf8 scripts/run_lora_rag_project_test.py --mode review
```

## 人工评测与归档停止点

全100对/200份文案；生成前冻结随机匿名A/B。正式口径完全沿用validation：matched_attribute_count、fluency_pass、factual_error_count；四项匿名偏好：标题、卖点结构、短详情自然度、整体专业度；五项诊断不替代正式指标。全文命中按原625个核心字段计数，正常同义允许，重复错误去重，源争议单列。事实可由完整源结构化属性支持，而非只按输入子集判错。

不预填正式人工判断，不把自动Grounding/literal coverage等同于人审。所有100行完成并确认后，先输出匿名汇总，再解盲，得到正式三指标、偏好、分品类及诊断case，冻结最终xlsx/hash和报告。

```powershell
& 'E:\Anaconda\python.exe' -B -X utf8 scripts/run_lora_rag_project_test.py --mode summarize
```

自动阶段完成后先停止交付工作簿；人工复核完成后才能形成最终人工test100指标。不会据结果修改组合或重跑。
