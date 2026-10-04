# P7：Base vs LoRA project_validation 冻结协议与复核操作

本协议先于第一次生成固定。范围为原冻结 project_validation 全量200件，每模型600个独立任务、200份组装文案。不训练、不按输出抽样、不调Prompt/数据/解码。人工复核未完成之前不产生正式人工指标。

用户明确允许Base/LoRA覆盖全部200件（含与RAG holdout同ID的商品）；不读取RAG holdout清单或归档，不运行RAG v2，不改变其 `frozen_not_executed` 状态。不读取 project_test。

## 固定输入和模型

- 输入：`data/processed/week1_v3/multimodal/validation.jsonl`，SHA-256 `dd2bc1d6c21567176d95f5414bfd9304f48979b3bb39ab0b038627b1f346af0b`。
- 共用P5 `build_instruction_text` 和system：`你是中文电商文案编辑，只能使用输入中的品类和可靠核心属性。`，版本 `lora_copy_templates_v1`。
- 每商品分别输入 `title`、`selling_points`、`short_description`；输入只有品类、可靠核心属性、任务。原标题只保存在复核元数据，绝不进入Prompt。
- 属性使用不变的P5值门控，结合已有validation-only静态字段审计，暂缓明确 `field_requires_review/field_conflict` 字段；身份/品类冲突不输入其属性。不使用RAG多值白名单规则，不运行检索。两侧输入完全相同。
- 所有200件保留，REVIEW/CONFLICT不排除。正式核心分母沿用 `select_core_attributes` 和冻结的 `configs/generation_evaluation.json`，在上述输入门控前计算，不因输入缺失或暂缓改分母。无输入/不足三字段仍原样生成，不补事实。
- 模型：Qwen/Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`，固定本地snapshot/offline。两侧NF4、double quant、BF16、冻结大层BF16/Norm FP32相同，推理eval/no_grad、cache=true、不启用gradient checkpointing。
- LoRA唯一额外模型组件：`artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter`（epoch3），权重SHA-256 `890a061812321ccf6bbf598b37617fc59dcde70d57042c4304c219a0f6740e31`。Base不加载adapter。训练配置不修改。

## 推理参数、配对、格式

共用 `configs/lora_project_validation_v1.json`，max_new_tokens=320，do_sample=true，temperature=0.7，top_p=0.9（沿用项目现有生成配置），top_k=20、repetition_penalty=1.05（固定基座generation_config），num_beams=1。完整resolved GenerationConfig在首次生成前另存两侧JSON，EOS等默认值也冻结。禁止输入截断；2048输入token上限只用于报错，不截断。

逐商品/任务seed：SHA256(`p7_lora_v1:42:{product_id}:{task_type}`)前8个hex转整数，模2147483647，不包含模型版本。每次调用重置Python、Torch CPU和CUDA随机seed。预先固定600个消息hash、token ID hash/preflight，确认可靠属性JSON经过tokenizer完整保留，两侧重新核对。

卖点解析接受恰好三个JSON字符串或三条编号/bullet条目；不为了强制JSON改P5 Prompt。格式失败、空输出或达到token上限只记录，不重试、不补写。人工正式评价始终依据全部原始输出，非仅解析成功的子集。完整文案直接串联三任务原文，同时保存独立原文与结构化解析结果。

## 速度与自动诊断

分别记录每个任务从generate开始至CUDA同步完成的延迟，包含该次GPU推理和采样，不包含预处理/tokenization/模型加载/解码；三任务总耗时是同商品三个延迟之和。进程wall time另报。首个调用含首次执行开销，不删除其样本；各任务平均/p50/p95同口径。不与旧一次调用Baseline直接归因比较。

字面命中、评价词、疑似数值、结构失败及token上限仅辅助线索，不产生人工三指标，不把自动FAIL当事实错误率。自动线索工作表默认隐藏，避免先看线索造成锚定。

## 人工复核操作（全200件，允许分批保存）

你需要做：

1. 打开 `reports/generation/lora/project_validation_v1/project_validation200_blind_review.xlsx`，先读“复核说明”。不要查看版本输出目录或匿名映射。
2. 进入“盲评成对复核”。每行一个商品、A/B两份文案，标题/卖点/短详情并列；隐藏的源属性、原核心清单、实际输入和完整文案列可展开。商品ID为文本，灰/锁定列不要改。
3. A、B各填黄色 `matched_attribute_count`：全文正确表达的核心字段数，0至本行原 `core_attribute_count`，同字段一次、正常同义/单位格式可认可；多值任一正确支持值可命中。数值或兼容范围失真不计正确表达。
4. A、B各填 `fluency_pass`：通顺1/不通顺0；允许标题短语和中性分条，不要求营销扩写。事实错误不自动导致不通顺，事实正确也不保证通顺。
5. A、B各填 `factual_error_count`：独立错误点数，同一错误重复只计一次。无依据评价/效果/场景、矛盾/失真、身份品类错误计入。直接低风险品类语义或支持的同义转述不计错误。对照完整源属性，不能只因为某源支持事实未进入输入就机械判错；原标题营销断言不能单独作为事实支持。
6. 四个成对选择分别填 `A更好 / B更好 / 持平`：标题表达质量、卖点结构、短详情自然度、整体电商专业度。看不出明确差异时选持平。
7. 可选填两侧五项诊断0/1：字段堆叠、机械模板化、无依据评价/效果/场景、数值/范围失真、身份/品类错误。空白不当作0。任何事实错误或诊断1须在对应notes写原文和依据。源争议另列，不机械等同于模型幻觉。
8. 填 `reviewer` 与 `review_confirmed=1`，保存后可继续下一批。全部200行的正式指标和四项成对比较齐全后告诉我，再汇总解盲。

映射在首次生成前按seed41007的独立SHA256排序固定，100件Base为A/100件LoRA为A；复核顺序另用独立namespace排序。不按输出、类别结果或优劣改匿名关系。映射另存JSON并保留在veryHidden页，隐藏不是加密。脚本完整检查填写、源/文案/映射不变，先保存匿名汇总，再应用映射形成Base/LoRA指标。人工结果不得由自动指标填充。

正式汇总：属性命中=命中总数/原核心总数；事实错误样本率=错误数>0商品数/200；平均事实错误=错误总数/200；通顺率=通过数/200。成对比较报告两模型胜/平的数量和200分母，诊断报告实际填写分母，不替代正式指标。

## 实际命令（首次冻结→Base→LoRA→工作簿；不自动重跑）

在 `E:\Projects\Baidu\Ecommerce-Content-Retrieval` 执行：

```powershell
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -X utf8 scripts/run_lora_project_validation.py --mode prepare
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -X utf8 scripts/run_lora_project_validation.py --mode generate --variant Base
& 'E:\Anaconda\envs\ecommerce-generation\python.exe' -X utf8 scripts/run_lora_project_validation.py --mode generate --variant LoRA
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation.py --mode workbook
```

工作簿使用既有base环境openpyxl，不在模型环境安装包。生成后停止，等待人工复核。未来完整复核确认后才能执行：

```powershell
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation.py --mode summarize
```

以上没有任何project_test命令。冻结产物在 `reports/generation/lora/project_validation_v1/`：`protocol_manifest.json`记录Git HEAD/dirty status及代码原始字节快照（不假装未提交代码等于HEAD）、config/数据/模型metadata/chat template/消息token hash、环境freeze；两侧outputs和review_manifest另封存hash。代码/config/Prompt变化立即报错，不由输出修改后续推理。不commit、不push。

数据职责：`lora_train`参数更新、`lora_train_dev`训练dev loss监控/选checkpoint；本轮`project_validation`验证已经冻结的候选；`project_test`最终正式测试，仍禁止读取。
