# 统一评测协议与结果使用说明

本文件更新原检索评测说明，作为最终统一入口。**冻结实验协议/人工标签/指标不修改，不重跑formal test或holdout**；原逐次protocol仍是精确执行证据。

## 1. 四个集合及freeze原则

| 名称 | 角色 | 是否更新模型参数 |
|---|---|---|
| lora_train | 原训练侧P5，1,206商品/3,618任务 | 是，仅已完成训练 |
| lora_train_dev | 同训练侧内部134商品/402任务 | 否，完整dev loss监控和epoch-end best选择 |
| project_validation | 原冻结200商品，模型候选效果验证 | 否；开发使用，不等于内部LoRA dev |
| project_test | 原冻结200商品；生成选固定100，检索选固定50查询 | 否；结果不回流调参 |

P5内部90/10不替代项目8:1:1，不另建test。训练选best仅用lora_train_dev，不用project_validation/test。P7两个模型均生成project_validation200商品×3task；人工在看质量前按8类、固定seed冻结quick32，不宣称200件全人审。组合development24是已观察开发样本，非未观察holdout。

test100在Baseline/RAG等历史阶段已被观察；组合为用户批准的追加固定test100对照，不是新的未观察盲测。test未用于LoRA参数更新不等于没有测试集复用/信息影响后续方案的风险。所有版本有独立输出、标签和manifest，结果无论好坏只归档；禁止看bad case改Prompt后重跑同一test。

RAG v2 32条generation-output holdout保持frozen_not_executed；P1已审计其静态字段，不能说字段分布未观察。本轮不打开清单、不运行；组合不是原P3晋级。

## 2. 生成正式人工三指标

按组装完整商品文案（title+selling_points+short_description）评分，三个task不是三个独立商品样本。

| 字段 | 冻结判断口径 | 汇总公式 |
|---|---|---|
| matched_attribute_count | 原冻结核心清单中，在全文任意位置正确表达的字段数量；同义/正常格式可计，失真不计；同字段重复不重复计 | sum(matched)/sum(core_attribute_count) |
| fluency_pass | 整份文案是否语法可读、逻辑自然；0/1。格式解析单独统计，事实错误单独判断 | 通过商品数 / 已完成人工商品数 |
| factual_error_count | 对照同商品完整、可可靠支持的源属性，统计矛盾或新增无来源事实的独立错误点；跨任务重复去重 | 错误样本率=count(error>0)/N；平均错误=sum(error)/N |

正式test100分母625，不能因P5/RAG撤回字段、源REVIEW/CONFLICT、空输入或输出解析失败删商品/改分母。自动字面命中不能直接复制为人审命中。一个源表达明确同时满足两个同义核心字段时可分别计字段命中，例如摇盖式对应垃圾桶类型/开合方式；不要求为了标签重复制造文字。

事实性不只按Top3或实际输入子集：未注入但完整可靠源明确支持的表述，不机械算幻觉；同时应记录是否遵守输入边界。源自身错误/歧义不自动算模型新增错误，不作外部商品真值宣称。unsupported evaluation/effect/scenario、数值/范围失真、身份错误按证据复核，不以单个关键词机械定罪。

允许不新增具体事实的品类直接语义/正常释义；没有来源的具体人群、场景、效果、因果强化仍计错。范围含/不含、兼容集合、数量、品牌/型号角色必须保持原义。比如源“塑料”不支持环保/耐用，源范围6小时含至12小时不含不支持12小时定值。边界例子须写证据及裁决，CT053“适用范围：家庭清洁”按本次用户最终确认保留严格评分。

**格式失败与事实错误独立**：区块提示泄漏、空身份字典、字段堆叠可导致结构/通顺不合格，但没有明确新增商品事实时不能直接算事实错误；0个事实错误也不表示输出合格。

来源实现：src/generation/evaluation.py、lora_validation_review.py:official_metrics及各正式review validator。旧Baseline四指标还有category_style_pass，后续三任务正式口径不临时增加该字段作为门槛。

## 3. RAG正式归因与自动诊断

RAG v1 formal记录：
- source_fact_quality_count：源问题，单列，不计模型事实错误。
- grounding_failure_count：提供的可信事实/约束被矛盾或失真表达。
- unsupported_generation_count：新增无来源参数/效果/场景。
- supported_paraphrase_count：有事实支持的释义，不计错。
- factual_error_count=grounding_failure_count+unsupported_generation_count，原正式配置不改。

Grounding PASS/FAIL是冻结validator结果，不能直接推事实错误率。检出的评价/效果/场景/数值词是人审线索；源有支持可能false positive，规则没检出也可能漏报。mandatory实际输入完整率只衡量输入供给，不是输出覆盖率。

分母须分开：静态category×field机会、实际存在核心事实、实际期待Mandatory事实、正式core_attribute_count。源missing不计Mandatory失败；任何token截断/缺失必须报错，不容许完整率100%的伪记录。

## 4. 公平对照、匿名偏好与评审来源

Base/LoRA：同输入、同task Prompt/chat template、同max tokens/decoding/逐商品逐任务seed，唯一模型变量为adapter加载。旧一调用Baseline不可直接对照三任务LoRA归因。

LoRA/LoRA+RAG：固定同一adapter及三任务设置，处理差异是四区上下文+必要system接口适配。不是纯检索单因素因果实验；control文本复用历史冻结输出，原旧标签不导入，配对评审重新进行。

四维匿名偏好：title_quality、selling_points_structure、short_description_naturalness、overall_ecommerce_professionalism，每项A更好/B更好/持平。先保存blind_summary.json，再使用封存映射解盲。各维度独立：结构规范不等于自然度更高，事实保守不自动等于标题质量更好；不能仅凭核心命中数量写表达偏好理由。

五项诊断：field_stacking、mechanical_template、unsupported_evaluation_effect_scenario、numeric_range_distortion、identity_category_error；不替代正式三指标。非零诊断/事实错误需写定位依据。所有行review_confirmed=1且reviewer非空才正式汇总，AI初审的确认标记不得自动替代用户最终确认。

本项目后期为ChatGPT初审+Codex修正+用户最终确认，不是两个独立真人评审；A/B隐去名称但展示的输入区块可能透露条件，不称严格双盲。Excel ID按文本保留，CSV使用UTF-8 BOM或XLSX，不编辑固定source/output/hidden mapping。

同一LoRA输出在两次评审语境可出现标签差异：LoRA-only正式为1%错误样本/5错误；组合配对control为2%/5错误。两者是不同冻结复核结果，不表明重新生成，不覆盖旧标签，也不能择优挑一个充当新实验对照。报告必须连同错误计数分组及边界裁决说明保留。

## 5. 检索标签和指标

validation24查询，每条Top20候选共480行人工判断；没有全库标注，因此只报告pooled Recall@10。test50在固定test200候选库，排除query自身，每条全24件同二级品类候选，50×24=1,200标签；跨二级品类按冻结标准记0。它是冻结评测设定下的完整分母，不等于在任意电商商品全库穷尽相关性。

| relevance_grade | 定义 |
|---|---|
| 2 | 品类正确且满足主要属性与用途 |
| 1 | 核心品类正确，部分满足且不违背最关键条件 |
| 0 | 品类错误或明确关键条件冲突 |

正相关threshold=1。每条查询：
- P@K=TopK正相关数/K。
- R@K=TopK正相关数/全部已定义候选中正相关数；validation只在pool中定义分母。
- MRR@K=首个正相关名次倒数，TopK无相关则0。
- DCG=sum((2^grade-1)/log2(rank+1))，NDCG=DCG/理想按全标注grade排序的DCG。
- 最终整体及品类取逐查询宏平均，不按商品数或标签行数加权。

实现src/retrieval/evaluation.py（pool）及formal_evaluation.py（test）；P/R为PRD核心，MRR/NDCG为冻结补充排序指标。纯跨模态策略要求T→I图片库、I→T文本库，正式test候选仅test200，不能偷偷换融合特征。

test查询v2三处、v3两处替换发生于首次正式指标前，经用户确认，理由为查询有效性/相关候选可得性；test200不变，旧48行标签另留。不要描述成原始完全随机且没有审核干预的独立样本；当前query/标签已冻结，不再换选或扩大后改正式指标。

## 6. 入口定位、hash关系和停止保护

| 用途 | 实际脚本 | 状态 |
|---|---|---|
| 生成Baseline | scripts/generate.py | FINAL/RECOMMENDED；仅新非正式输出可演示，默认会覆盖同名文件 |
| 生成Baseline/RAG人工指标 | scripts/evaluate_generation.py | FINAL；config选择具体口径；本轮不执行写正式metrics |
| RAG v1 formal生成 | scripts/run_rag_formal_test.py | FINAL，已完成单次test100，禁止重跑 |
| 三任务Base/LoRA test | scripts/run_lora_project_test.py | FINAL，prepare/generate/archive/summarize均是有状态工作流，已完成 |
| LoRA+RAG追加test | scripts/run_lora_rag_project_test.py | FINAL，已完成生成与汇总，不再次运行 |
| validation人审/quick32 | scripts/review_lora_project_validation*.py | EXPERIMENT，开发证据 |
| 检索pool指标 | scripts/evaluate_retrieval.py | EXPERIMENT，validation开发；不冒充test |
| 检索test50 | scripts/evaluate_test_retrieval.py | FINAL，已完成，禁止重跑 |
| rerank test50 | scripts/run_retrieval_rerank_formal_test.py | FINAL，已完成，禁止重跑 |
| 静态交付核对 | scripts/audit_delivery.py | 第一轮无模型静态核对；批准缓存清理后的保护检查见reports/delivery/cache_cleanup_verification.json，不重写原baseline |

追溯顺序：最终报告 → 最终metrics/人工标签 → final manifest/checksums → protocol/config/真实代码字节 → 数据/adapter/model元数据/环境hash。当前Git HEAD不能代表未提交文件；独立工作区hash与代码快照必须同时交付。

旧automatic报告中的human pending、配置pending状态是历史快照，不能为了可读性改冻文件。最后finalization manifest给出完成人审状态。已归档结果只读，禁止通过“重算评测”覆盖结果；本轮静态统计和hash核对不调用模型或评测入口。
