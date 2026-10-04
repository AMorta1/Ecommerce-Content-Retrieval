# P1-RAG 只读否定约束与 Grounding Validator 验证报告

## 范围与不变量

- 仅使用 validation；未读取或运行 test100 生成结果。
- 继续使用全局共享 Top-3、恰好 3 个卖点、同一 Qwen 模型及 revision、原量化和解码参数。
- Validator 仅检测和记录，不删除、不改写、不重试生成，也不替代正式人工评测。
- 第三批新样本排除了此前参与 Prompt 调整和验证的 48 件商品，按 8 个二级品类各抽取 3 件。

## 只读否定约束

事实选择器在质量门控后把以下合格值识别为否定约束：

- 精确值：`无`、`否`、`不支持`、`有线`、`没有`、`未配备`；
- 包含“不支持”的合格值。

处理规则：

1. 仅接受同一 `product_id`、`fact_role=task`、`quality_status=eligible` 的事实；
2. `REVIEW` 暂缓字段和 `CONFLICT` 阻断字段不会进入约束区；
3. 否定事实从内容候选中移出，不占全局 Top-3；
4. Prompt 使用独立的“只读否定约束”区；
5. 约束只禁止相反内容，不要求主动写入标题、卖点或短详情。

第三批 24 件商品中，4 件共注入 6 条只读否定约束。没有发现与这些约束相反的生成内容。

## Prompt 通用约束

`rag_v6` 不再针对 validation 的具体词语增加错误示例，只规定：

- 身份信息用于识别商品；
- Top-3 是唯一可主动写入的内容事实；
- 否定区只用于禁止相反内容；
- 允许不增加新事实的品类直接语义释义；
- 禁止从品类或字段进一步推导具体场所、人群、对象、效果、性能或相对评价；
- 内容事实不足时用合格品类、品牌、型号补足三个卖点。

## Validator v1

Validator 输出逐条 finding，并区分：

- `supported_paraphrase`：品类基本用途、分类标签组合或明确事实字段的直接释义；
- `unsupported_usage_scenario`：没有品类直接语义或明确场景事实支持的具体场所、对象或用途；
- `unsupported_evaluative_claim`：没有源事实或明确规则支持的相对、效果或质量评价；
- `grounding_failure`：与只读否定约束或带方向的内容事实直接冲突。

每条结果同时记录位置、文本片段、判断原因和疑似误报标记。Validator 报告明确记录：

- `mode=detect_and_record_only`；
- `output_modified=false`；
- `regeneration_triggered=false`；
- `replaces_human_evaluation=false`。

## Validator 误报和漏报检查

开发初检发现并修正：

- 3 个明显误报：拖把日常清洁、分类标签组合、头戴式的直接释义；
- 3 个明显漏报样本：`6小时以下→6小时以上`、`优质/方便`、`优质选择`。

修正后重新对同一批保存输出进行确定性验证。最终 validator 与 Codex 逐条初审在本批 24 条的生成错误判断上一致：

- 明显误报：0；
- 明显漏报：0；
- 另有 1 条源事实质量 REVIEW，不属于 Unsupported Generation，因此 validator 未判为生成错误。

该结果只说明当前样本和当前规则中没有发现明显误报/漏报，不能替代正式人工评测，也不能证明规则覆盖所有中文表达。

模型运行日志 `validation_holdout2_rag_v6.json` 保留运行当时的首轮嵌入式 validator 结果，没有覆盖。修正规则后的权威只读复核结果单独保存在 `validation_holdout2_rag_v6_validator_v1_final.json`，其中记录了原日志路径和 SHA256；模型原始输出没有改变或重跑。

## 历史场景 Bad Case 重新归类

按照新口径：

- `585183140226` 的“适用于家居日用”归为 `supported_paraphrase`；
- `613971147597` 的“适用于家居收纳”归为 `supported_paraphrase`；
- `617300397884` 的“适用于家居日用”归为 `supported_paraphrase`；
- `600307012365`、`605201240573` 的“适用于家居日用收纳”归为 `supported_paraphrase`；
- `613342790955` 的“适合多种设备充电需求”仍为 `unsupported_usage_scenario`，因为增加了“多种设备”对象范围。

历史 v3 报告重判为 24 PASS，并记录 3 次 `supported_paraphrase`。上一批 v5 新样本重判为 20 PASS、4 FAIL；4 条 FAIL 均为真实评价或具体对象扩写。

## 第三批新 validation 结果

### 样本与结构

- 商品：24 件，与此前 48 件零重叠；
- 品类：8 个二级品类各 3 件；
- RAG 成功解析：24/24；
- Baseline 成功解析：23/24；
- RAG 均保持恰好 3 个卖点。

### Validator

- PASS：17；
- FAIL：7；
- `supported_paraphrase`：6 个 finding；
- `unsupported_usage_scenario`：3 个 finding；
- `unsupported_evaluative_claim`：9 个 finding；
- `grounding_failure`：1 个 finding。

### Codex 逐条初审

- PASS：16；
- REVIEW：1；
- FAIL：7。

REVIEW 商品 `601447163749` 的光学分辨率为占位式源值 `其他/other`。输出忠实使用了该事实，因此不属于模型 Unsupported Generation，但说明当前事实单元的占位值过滤仍有缺口。

## 剩余 Bad Case

- `534591817203`：源事实“6小时以下”，短详情生成“6小时以上”，属于 `grounding_failure`；
- `621946386369`：新增“方便”“优质”；
- `604712328413`、`564110705919`：数值容量扩写为“大容量”；
- `616853015046`：新增“优质选择”；
- `613623155562`：新增电脑办公场景和“稳定可靠”；
- `594830440707`：新增电脑游戏和办公场景；
- `601447163749`：源事实占位值质量 REVIEW。

## 冻结判断

从工程和实验设计角度，当前方案已经具备冻结为首版 `rag_v1 formal` 候选并进入一次性 test100 评测的条件：

- 模型、revision、解码、Top-3、Prompt、负约束和 validator 均可追溯；
- 新样本未参与 Prompt 调整，否定约束没有被违反；
- Validator 的检测结果已与逐条初审对照；
- 剩余错误会被保留并在正式评测中计入，不会自动修复或重试。

该判断不表示 validation 已无错误。当前新样本仍有 7/24 明确生成失败和 1/24 源事实 REVIEW。继续针对这些样本追加 Prompt 词语补丁会增加 validation 过拟合风险，因此建议在用户确认后冻结当前通用方案，而不是继续逐词调 Prompt。

正式冻结前仍需生成不可变的 `rag_v1 formal` 配置和 manifest，并明确记录占位源事实的错误归因方式；本阶段未创建正式配置，也未运行 test100。
