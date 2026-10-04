# LoRA v1 最终 project_test100 协议

用户因项目周期明确接受validation trade-off，不再迭代LoRA v1，批准一次正式测试。本协议范围是原冻结test100全部100件，不是原project_test全部200件，不重新抽样。每模型100件×3任务=300次调用，共600次；完整100对盲评，不按输出挑人工子集。

冻结epoch3/checkpoint-step-001359及其adapter、训练config、Prompt/chat template、输入门控、max tokens、decoding、逐商品/任务seed和三任务推理路径。所有推理设置继承P7冻结协议。Base/LoRA唯一模型变量为adapter有无，输出隔离保存。禁止参数更新、生成重试/续跑、根据test调整模型或Prompt，禁止RAG holdout归档和RAG v2运行。

质量输入继续沿用P5可靠核心属性选择，再按既有源审计暂缓标记字段；身份/品类冲突整件撤回输入属性但保留商品及原核心分母。prepare先将原validation审计元数据转换结果与已冻结P7输入逐条核对，确认门控实现等价；不以test内容决定规则。test100的源质量审计是既有只读文件，不读取任何旧生成结果或人工标签作为输入。

第一次模型调用前冻结config、源test100及其父project_test hash、原抽样manifest、全部100件ID/品类/顺序、匿名映射、可靠输入、三任务消息/hash/seed/token preflight、代码原始字节与Git HEAD/dirty status、环境freeze、基座revision/本地snapshot/模型和tokenizer元数据、best adapter权重/config hash。Git未提交时记录真实代码快照，不将HEAD冒充当前工作区代码。

原test100曾被Baseline/RAG使用：本轮是LoRA v1首次且仅一次正式配对测试，并非从未观察的测试集盲测。旧单调用Baseline结果不作为本轮公平对照；生成新的同三任务Base。每版本目录存在即拒绝再执行，异常保留部分输出并停止，禁止自动重跑。

## 执行命令

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode prepare
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode generate --variant Base --confirm-formal-test100
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode generate --variant LoRA --confirm-formal-test100
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode archive
```

archive生成完整100件 `project_test100_blind_review.xlsx`，三个任务同一行，原属性/核心清单/可靠输入及源争议同一行。100对映射在生成前冻结，50件Base-as-A/50件LoRA-as-A；隐藏映射veryHidden，评审完成前不查看映射或版本标识输出。黄色列评分留空，不用自动指标或validation标签预填。

人工口径复用validation：全文正确命中核心字段一次、正常同义/格式允许、多值任一可靠值；失真表达不算正确命中。通顺与事实正确独立。独立事实错误跨任务重复计一次；对照完整源属性，原标题不单独支持营销断言，源争议写notes不改分母。填写matched_attribute_count、fluency_pass、factual_error_count，四个A/B偏好、五项诊断、证据notes、reviewer、review_confirmed=1。固定结构本身不自动算机械模板。

自动指标包括三任务解析、组装结构、字面覆盖、每任务及总延迟、空输出、token上限、既有疑似评价词/数值规则；另提供通用场景/效果疑似词及区间边界字面缺失线索，但不修改既有validator或正式人审口径。这些不等于人工事实错误，分品类自动统计也明确标注。全部原始失败输出保留，不剔除或重试。

完成100对最终人工确认后才运行：

```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -B -X utf8 scripts/run_lora_project_test.py --mode summarize
```

先匿名汇总再解盲，形成正式三指标、偏好、分品类、人工确认bad case和最终manifest。人工未完成时报告明确human_review_pending，人工指标null，不编造结果。自动阶段完成后停止等待复核；本次test结束后不再基于其结果改变模型、Prompt或参数。
