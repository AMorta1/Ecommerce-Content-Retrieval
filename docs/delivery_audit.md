# 最终交付前文档与仓库审计

第一轮根据Leader的三项交付要求整理：数据说明、技术沉淀、可读性，只更新说明和新增静态审计工具/报告，没有清理文件。后续用户已批准仅删除49个缓存、ARCHIVE原位保留、仅完善文档索引，执行记录见第9节；两阶段均不安装依赖、改算法/数据/Prompt/config/正式结果，不训练或运行test/holdout，不commit/push。

## 1. 文档盘点及更新/合并/新建决策

本节表格记录第一轮审计决策。用户后续确认将文档职责拆分：README恢复为项目使用与导航入口，新增[docs/project_brief.md](project_brief.md)作为正式成果简报；详细指标从README移入简报，冻结报告不修改。以下第一轮记录不代表当前仍将README作为简报。

| 现有内容 | 修改前覆盖与缺口 | 直接更新 | 合并建议 | 新建？ |
|---|---|---|---|---|
| README.md | Week1目标/流程，多段历史指标；缺最终RAG/LoRA/组合与rerank状态 | 已更新为项目简报、最终结果和5–10分钟接手索引 | 不另造项目README或重复简报 | 否 |
| docs/data.md | 原始清洗/图片/8:1:1/去重已有；缺P5、字段体系、质量门控、各阶段分母 | 已补齐正式《数据处理说明文档》 | 整合分散规则说明，原统计/报告只引用不移动 | 否 |
| docs/generation.md | 单调用Baseline历史流程、旧指标占主体；缺LoRA/RAG全技术细节 | 已更新为生成/RAG/QLoRA统一入口 | 用链接接入已有protocol与正式报告，不修改冻结材料 | 否 |
| docs/retrieval.md | ERNIE-ViL/索引/环境已有；仍写rerank未完成，未明确特征位置 | 已补当前源码/预处理观察、正式rerank和接口缺口 | 不另造单一ERNIE-ViL重复文档 | 否 |
| docs/evaluation.md | 主要为retrieval pool/test标签流程；无完整三任务/匿名/事实性说明 | 已更新统一评测入口 | 原generation评审口径及protocol通过链接统合 | 否 |
| docs/lora_*protocol.md、quick32说明 | P7/test/组合冻结实验协议，详细但分散 | 不改，冻结证据 | 不物理合并；统一入口解释对应阶段 | 否 |
| reports/formal_baseline_report、RAG v1、rerank及LoRA/组合最终报告 | 正式状态/结果/hash充分，不是整个项目简报 | 不改，冻结证据 | 报告不合并、不复制成新“最终v2” | 否 |
| 多个final_test_report / final_test_analysis / final_review_report | 同实验有机器详细与人读解读，存在覆盖重合但hash链不同 | 不改 | LoRA展示final_test_analysis；组合展示final_review_report；其余作追溯入口 | 否 |
| environment_freeze、requirements与manifest | 模型环境已散在报告；依赖声明未完整包含LoRA/工作簿工具 | 文档说明差异，不改依赖或环境 | 分环境索引，不把所有freeze强行合并到通用环境 | 否 |
| Colab迁移指南.md、Plan.md、handoff.md、Week2.md | 迁移/历史计划/交接，部分早于当前状态 | 不改 | 历史辅助阅读；不当最终事实源，不重复放主入口 | 否 |
| 本交付审计与逐文件inventory | 未发现能承载四类全仓分类、入口/缺口及整理审批的既有文档 | 本文件+reports/delivery新建 | 与运行说明分工，不重复技术细节 | 是，仅缺失的审计 |

更新前检查这些五份说明没有被列为冻结manifest的path/hash文件条目；有些历史Git/diff字符串提及README不等于文件锁定。本轮protected_files_before记录其余非缓存资产，完成后protection_verification证明未改冻结输入/代码/config/结果。

## 2. 本轮新增证据与范围

- [repository_inventory.csv](../reports/delivery/repository_inventory.csv)：逐文件path、四分类、理由、入口角色、bytes、Git跟踪、SHA及冻结引用。
- [inventory_summary.json](../reports/delivery/inventory_summary.json)：分类文件数/总字节及Git HEAD。
- [data_statistics.json](../reports/delivery/data_statistics.json)：真实JSONL的split/品类/task计数、ID隔离及原统计依据。
- [code_readability_findings.json](../reports/delivery/code_readability_findings.json)：静态AST检查docstring/本机路径/模糊命名，不执行代码。
- [runtime_observations.json](../reports/delivery/runtime_observations.json)：当前安装的PaddleNLP源码及缓存preprocessor配置hash，不冒充历史运行记录。
- [protected_files_before.json](../reports/delivery/protected_files_before.json) / [protection_verification.json](../reports/delivery/protection_verification.json)：整理前后未修改资产校验。

覆盖整个项目实际文件，包括Git忽略的data、artifacts及缓存；不包含Git内部对象，不自包含reports/delivery以避免递归hash。仓库外原始大JSON、基座缓存及本机环境不是清理范围。本轮不重扫631万条原始数据，原始量与白名单统计来自既有census，现有实验JSONL做只读实统计。holdout文件只登记size/mtime，不打开内容；状态引用已有阶段报告。

## 3. 最终推荐入口与用途

FINAL表示已确认功能/正式实验入口，**不表示现在可以把已完成train/test重跑**。没有通用入口的地方明确标缺口。

| 功能 | 入口 / config / 核心函数 | 标记与运行边界 |
|---|---|---|
| generation Baseline | scripts/generate.py；configs/generation.json；QwenGenerator/build_messages | FINAL/RECOMMENDED；新非正式generation_input，新输出文件，勿覆盖 |
| LoRA指令数据构造 | scripts/build_lora_instruction_data.py；configs/lora_instruction_data_v1.json；lora_data.py | FINAL；P5已冻结，本轮不重建 |
| LoRA training | scripts/train_lora.py；configs/lora_qlora_train_v1.json；lora_training.py:run | FINAL；不是smoke trainer，已完成3epoch，不启动 |
| LoRA inference | lora_validation.py:generate / lora_formal_test.py:generate中的固定base+PeftModel加载链 | 实现存在；通用单商品CLI缺失，不能把test脚本当服务 |
| adapter独立验收 | scripts/verify_lora_v1_best.py | 工程验收已完成，拒绝覆盖已有报告，不是通用LoRA Demo |
| RAG generation | scripts/run_rag_formal_test.py；generation_rag_formal_v1.json；qwen.py:build_rag_messages | FINAL冻结单调用v1；四区v2为EXPERIMENT；通用RAG CLI缺失 |
| LoRA+RAG | scripts/run_lora_rag_project_test.py；lora_rag_project_test100_v1.json | FINAL追加固定test；通用单商品CLI缺失 |
| ERNIE-ViL embedding | scripts/build_index.py / build_text_features.py；retrieval.json；ErnieVilEmbedder | FINAL/RECOMMENDED；已有冻结附件，本轮不重建 |
| retrieval普通推理 | scripts/search.py --text/--image | FINAL/RECOMMENDED；纯跨模态Baseline，不含正式rerank |
| retrieval rerank | src/retrieval/rerank.py；retrieval_rerank_formal_v1.json；run_retrieval_rerank_formal_test.py | FINAL公式/正式入口；任意query在线集成缺失 |
| retrieval evaluation | evaluate_test_retrieval.py / run_retrieval_rerank_formal_test.py；formal_evaluation.py | FINAL已完成；evaluate_retrieval.py只validation pool，EXPERIMENT |
| generation evaluation | evaluate_generation.py；lora_formal_review.py / lora_rag_project_test.py:summarize | FINAL口径，已有结果不覆盖；人工填表和解盲是有状态流程 |
| 静态交付检查 | scripts/audit_delivery.py --verify | 第一轮静态检查；清理后原baseline存在已批准差异，当前结果见cache_cleanup_verification.json |

以下按同名组识别历史入口，无需重命名就能读懂：

- SMOKE：run_qlora_smoke.py、run_qlora_low_memory_smoke.py、smoke_test_retrieval.py；对应config/adapter不得作正式训练初始化。
- EXPERIMENT：run_rag_validation.py、run_lora_project_validation.py、run_lora_rag_integration.py、tune_retrieval_rerank.py、analyze_retrieval_bad_cases.py和quick32评审工具。
- LEGACY配置：generation_greedy_baseline、generation_rag_validation_v1–v8及rag_fact_policy_v1/v2；v3用于冻结RAG v1，v4用于四区/组合，不是简单“v4取代所有v3”。
- 一次性归档：import_lora_*ai_review、finalize_lora_*、archive_lora_project_validation；历史源文件绝对路径/hash是有意固定的审计边界，不当通用运行工具。
- 没有发现需要命名为DEBUG的正式入口；不能因为脚本短或单独调用就认定可删除。

## 4. KEEP_FINAL / KEEP_REPRODUCIBILITY / ARCHIVE / DELETE_CANDIDATE

完整逐项理由在inventory，不仅给目录级结论；以下为可读摘要。四类是交付展示/建议分类，不是执行动作。任何冻结引用需保留原路径，不能只按分类移动。

| 分类 | 代表范围 | 为什么 / 当前动作 |
|---|---|---|
| KEEP_FINAL | README、四份技术/数据/评测说明、冻结protocol、src、tests、正式入口与config、正式结论/metrics | 普通接手者应先找到；保留，冻文件不改 |
| KEEP_REPRODUCIBILITY | 数据/图片、正式adapter/checkpoint、特征/index、原始task/product输出、人工原表/备份、hash/manifest、环境与代码快照 | 结果追溯和复现必需；大文件单独交付，不提交普通Git |
| ARCHIVE | smoke及retry、P6.2工程状态、P1/P2/P2.5/P2.6开发候选、development24、P5/P7过程/抽查/填写报告、旧Prompt/config、导入与finalize脚本、Colab/历史交接 | 具有历史或失败保护价值，但不应误当推荐入口；先在索引标historical，原位保留 |
| DELETE_CANDIDATE | __pycache__ / .pyc等可再生缓存 | 无独立复现价值，等用户确认才清；若hash绑定则改归KEEP_REPRODUCIBILITY |

ARCHIVE中的拒绝候选、失败retry能说明OOM/精度问题及为什么选当前工程方案，不能一概“只保留成功版”。正式人工表的initial_blank/pre_ai_import/source/user_completed看似重复，但承担不同hash与用户确认链，均不是DELETE_CANDIDATE。未确认无引用的临时副本不擅自认定可删。

P1/P2/P5/P6/P7不是全目录自动归档：P5最终训练数据/config、P6正式trainer、P7被后续test复用的基础模板/协议必须保留可定位；阶段最终结论也保留主索引。inventory的frozen_reference_count用于提醒，不是完备动态import调用图，0不能推定无用。

## 5. 代码可读性审计及建议（不改冻代码）

### 名称与重复

静态检查没有发现test2.py、tmp.py、new.py、final_v2_new.py、run3.py这类模糊命名。主体代码不以xxx_v1/v2/v3重复复制；版本集中在configs和独立实验scope模块。lora_validation、lora_formal_test、lora_rag_integration、lora_rag_project_test职责不同，并非只留最后一个就安全。

建议未来将scripts/import_*与finalize_*在文档标“一次性归档工具”；run_*project_test标“已封存正式实验”。**不建议当前直接改名**，改名会打断protocol file hash与保留代码快照对应关系。configs中的status=pending可能只是冻结时状态；最终manifest优先，不直接改config。

### 注释与接口契约

rag.py、qlora_memory.py、ernie_vil.py、rerank.py已有有用docstring；lora_training.py:run、lora_validation.py:build_inputs/generate、lora_rag_project_test.py:prepare/generate/summarize及lora_data.py多函数缺docstring。静态定位见code_readability_findings（行号可复查）。这是维护缺口，不是算法错误。

本轮用docs/generation、data、evaluation记录这些函数的输入/输出、状态、config和保护边界。以后如果允许改冻源码，应先归档旧字节，再添加契约说明或非冻结外层包装，不能改源码后将旧manifest当仍有效。无需给每行注释。

建议重点契约：build_inputs输入原split+policy/audit，输出原正式分母与门控事实分离的sample；run输入train/dev/config，输出adapter+可恢复状态/日志；generate输入冻结protocol，输出三个task原文及组装；summarize输入已确认workbook，先匿名后解盲，写一次最终metrics。

### 硬编码与路径

正式大部分数据/model/output由config和仓库ROOT解析；但scope、允许路径、hash、revision、seed namespace和正式输出名在code中有固定校验，这是防止误跑的冻结保护，不能为了“全可配置”解除。

实际本机绝对路径包括AI导入脚本E:/WebDownload附件，配置中observed_snapshot_path、环境/报告的E:/Anaconda、历史Colab指南。它们不是通用业务路径，交付时标明观察/一次性审计用途。静态扫描按AST字符串识别磁盘路径，避免把https://误报成S盘路径。

seed/LR/rank/rerank权重等在最终config与guard代码交叉核验，没有在本轮集中提取/换值。generic CLI可再使用config，而formal实验guard保持固定；现有公开推理适配需要另行授权，不能借收尾重构改变冻结参数。

## 6. 第一轮README修改前问题及解决程度

原README停留Week1，读者易误以为LoRA/rerank尚未完成，旧/新Baseline指标段落混用，长时间线遮挡推荐入口，默认smoke输出可覆盖，没有解释数据附件/adapter/内部dev命名。

第一轮改成：目标与两模块 → 最终结果与PRD未达标项 → 数据边界 → 阅读顺序 → 目录 → 双环境 → 可用非正式演示 → 正式结果定位 → 未完成事项。用户后续要求README与简报分离，目前README提供标准项目使用与导航，成果结果进入project_brief.md；训练/test命令不作为快速启动执行链，历史config状态不被误读。

仍不能仅发Git后“开箱即用”：未跟踪的P5/P6/P7代码与报告、大数据/图像/adapter/index需另附；requirements不完整，历史不可变权重revision缺失，通用LoRA/组合及在线rerank入口未有。README如实写明，不掩盖阻断。

## 7. 建议最终目录与整理策略

首选**保留现有路径，通过索引逻辑归档**，无需新建多套vfinal目录。

~~~text
README.md                      项目使用 + 快速导航
docs/project_brief.md          正式项目简报 + 结果与目标完成情况
docs/{data,generation,retrieval,evaluation}.md
docs/delivery_audit.md          清理审批与维护缺口
docs/*protocol*.md             冻结实验协议原位
configs/                       原位，FINAL/EXPERIMENT由索引标识
scripts/                       原位，入口角色由索引标识
src/ + tests/                  正式实现与回归保护
reports/formal_baseline_*       冻结基准
reports/generation/{rag,lora,lora_rag}/
reports/retrieval/              正式结果与原始证据原位
reports/delivery/              本轮静态审计，独立于实验manifest
data/ + artifacts/             独立大文件交付附件
~~~

如果未来获准物理归档，可以用reports/archive、artifacts/archive、scripts/archive作为**新快照索引/副本包**，先建立old→new映射、检查所有引用和hash；冻结原相对路径仍需保留或提供经过验证的独立复现副本。不能直接移动然后修改旧manifest，使它看似原始实验记录。当前无需强制加这三个目录。

第一轮建议小型说明/代码/配置/测试/报告提交Git，大文件独立交付。用户随后明确批准将整个artifacts加入Git跟踪，目录已暂存，.gitattributes禁止换行转换，冻结文件hash不变；数据和基座缓存仍不随Git提供，checksum不替代实际附件。inventory记录的是第一轮跟踪状态，不代表这次授权后的Git状态；尚未commit/push。

## 8. 需要人工确认的事项

1. 原始数据的正式来源名称、许可及可向Leader交付范围；不能自行补公开dataset链接。
2. 历史Qwen Baseline与ERNIE-ViL不可变revision、原始源全文件SHA、历史processor hash缺可靠记录，是否存在仓库外材料可补证。
3. Gradio Demo、PPT或独立端到端性能验证是否在仓库外已有；当前仓库未找到，不写已完成。
4. 是否需要另行补通用单商品LoRA/RAG/组合推理和search rerank接口；只是交付易用性方案，本轮不实现，不做效果实验。
5. 是否冻结补充依赖声明（PEFT、openpyxl/Pillow分环境差异）；本轮不改requirements，不升级包。
6. 最终代码/报告提交范围、大文件备份/发送途径；best adapter、checkpoint、data与indices需实物交付。
7. 是否同意先只清理未冻结引用的缓存；是否接受原位逻辑归档而非移动历史文件。
8. 接受最终目标未全面达标及已观察test复用、非独立双人标注的限制披露；不得删不利结果或改分母。

第一轮说明和审计完成后停止；用户随后确认了第9节的有限清理范围，未批准其他cleanup/rename/refactor。

## 9. 用户已确认的有限清理执行记录

范围：仅删除初轮inventory列出的49个DELETE_CANDIDATE缓存文件；196个ARCHIVE文件全部原位保留；其余仅完善文档/索引。不移动、重命名、重构或新增推理CLI，不改依赖、数据、模型、参数、Prompt/config或冻结实验结果，不commit/push。

删除前重新核对了：仓库内绝对路径、Git未跟踪、SHA与原inventory一致、manifest引用为空；并搜索项目代码/配置/报告/文档中的具体路径或hash引用。排除符号链接/reparse point，仅逐文件Remove-Item，不递归删目录。

删除49个文件，共595,319 bytes，均位于__pycache__或.pytest_cache。空缓存目录不扩展清理。第一次操作因沙箱访问限制在删除7个后停止，经受控权限提升完成剩余42个；没有修改ACL或扩大目标范围。

- [cache_cleanup_manifest.json](../reports/delivery/cache_cleanup_manifest.json)：明确的删除文件清单与操作记录。
- [cache_cleanup_verification.json](../reports/delivery/cache_cleanup_verification.json)：清理后非目标文件hash、ARCHIVE原位、Git HEAD和文档链接检查。
- 原repository_inventory、protected_files_before和protection_verification是第一轮清理前证据，保留原样。不能把旧protection_verification误读为清理后检查，也不能重写旧baseline来掩盖已批准变化。

原scripts/audit_delivery.py --verify对清理前baseline严格比较，当前会因已批准删除的6个pytest缓存与.gitignore变化报差异；本次以独立清理验证说明这些例外，不解除原保护逻辑。

清理后分类：KEEP_FINAL 129、KEEP_REPRODUCIBILITY 2,212、ARCHIVE 196、DELETE_CANDIDATE 0，共2,537个项目文件（不含.git内部对象及reports/delivery自身审计产物）。Colab迁移指南.md本地保留，只新增Git忽略规则。

删除不进入回收站；这些缓存可以由Python/pytest再生成，不是历史原字节备份。若需恢复原字节，当前没有缓存备份；不影响源代码、数据、权重或正式评测证据。任何进一步删除或物理归档仍需新的明确授权。
