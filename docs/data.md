# 数据处理说明文档

这是 Leader 要求的正式数据入口，补充原 docs/data.md，不另建同内容文件。统计来自当前文件和冻结报告，新增只读统计见[交付数据核对](../reports/delivery/data_statistics.json)。本说明不改变数据、split、清洗代码或实验结果。

## 1. 来源、版本与隔离边界

来源是 PRD 配套电商数据附件；当前工程材料中未找到可靠的公开数据集名称、下载地址、授权及许可记录，需要人工确认。实际原始文件由 configs/data.json管理，位于仓库上一级：

| 文件 | 原始记录数 | 用途 |
|---|---:|---|
| product1m_product5m_id_label.json | 6,313,067 | 商品总表，约5.79GB，流式读取 |
| product1m_product5m_train_id_label.json | 4,418,550 | 来源训练ID/标签及来源追溯 |
| product1m_product5m_test_id_label.json | 1,341,932 | 来源测试ID/标签及来源追溯 |

来源文件名不代表本项目实验split。252条抽中记录的original_split_unknown被保留；本项目采用新的固定分层split，不能冒充原数据集官方划分。[原始处理证据](../reports/data/preprocessing.json)只有bytes/mtime/records，不是完整原始文件SHA锁；此历史缺口不回填推测值。

| 项目版本 / 口径 | 真实路径 | 数量 / 角色 |
|---|---|---|
| week1_v3图文商品 | data/processed/week1_v3/multimodal/products.jsonl | 1,992件，检索库 |
| project train | data/processed/week1_v3/multimodal/train.jsonl | 1,592件，初始训练侧 |
| project_validation | data/processed/week1_v3/multimodal/validation.jsonl | 200件，项目级开发/候选验证 |
| project_test | data/processed/week1_v3/multimodal/test.jsonl | 200件，已冻结正式测试侧 |
| week2_train_v1 | data/processed/week2_train_v1/train.jsonl | 1,559件，仅从原train排除33件近重复 |
| P5可用商品 | data/processed/lora_instruction_data_v1/products.jsonl | 1,340件，4,020条三任务 |
| lora_train | data/processed/lora_instruction_data_v1/train.jsonl | 1,206件，3,618任务，参数更新 |
| lora_train_dev | data/processed/lora_instruction_data_v1/validation.jsonl | 134件，402任务，只监控loss/best选择 |
| generation正式样本 | data/processed/week1_v3/generation_evaluation_samples.jsonl | 原test中固定100件、625项核心分母 |
| retrieval正式查询 | configs/retrieval_test_queries.jsonl | 原test200中固定50条查询 |

P5文件名validation.jsonl和记录split=validation只是内部 lora_train_dev，不是project_validation。90/10仅在1,340件原训练侧内部按商品分组、八类分层SHA256排序、seed42划分；同商品三任务共置，不创建test，不重抽或修改项目validation/test。

project_validation用于冻结候选后验证效果，不用于本次LoRA参数更新或best选择；project_test不参与训练/模板设计/early stopping。当前交付核对：lora_train与lora_train_dev、二者与project_validation/project_test的商品ID交集均为0。ID无交叉不证明所有语义近重复完全消失；冻结validation/test之间仍有2个同品牌型号组，仅记录不移动。

原RAG v2的32条generation-output holdout保持frozen_not_executed，状态来自已有阶段归档；本轮不打开其清单。P1曾观察全部validation的静态字段，不能称字段分布完全未观察。test100曾用于多个实验，属于已观察固定测试数据。

## 2. 品类分布：不同阶段不能混为一个分母

原始数量是全量census中白名单标签命中数，不是总表全部商品按人工品类标注的数量。清洗候选为抽样前数量；每类随后只选250件。

| 一级/二级品类 | 原始白名单 | 清洗候选 | v3 train | project_validation | project_test | 图文总数 | week2 train |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3C / 耳机 | 9,667 | 9,615 | 200 | 25 | 25 | 250 | 195 |
| 3C / 键盘 | 2,354 | 1,649 | 197 | 25 | 25 | 247 | 184 |
| 3C / 鼠标 | 6,004 | 5,878 | 199 | 25 | 25 | 249 | 194 |
| 3C / 移动电源 | 3,330 | 3,312 | 199 | 25 | 25 | 249 | 192 |
| 家居 / 保温杯 | 3,010 | 2,943 | 200 | 25 | 25 | 250 | 200 |
| 家居 / 收纳箱 | 996 | 995 | 199 | 25 | 25 | 249 | 199 |
| 家居 / 拖把 | 2,928 | 2,844 | 198 | 25 | 25 | 248 | 195 |
| 家居 / 垃圾桶 | 1,617 | 1,203 | 200 | 25 | 25 | 250 | 200 |
| 合计 | 29,906 | 28,439 | 1,592 | 200 | 200 | 1,992 | 1,559 |

图文一级品类：3C 995、家居997；原train为795/797，项目validation和test各100/100；week2 train为765/794。P5一级品类为3C675、家居665；lora_train为608/598，lora_train_dev各67件。任务记录数等于商品数×3。

| 二级品类 | P5可用商品 | lora_train商品 / 任务 | lora_train_dev商品 / 任务 | 全部任务 |
|---|---:|---:|---:|---:|
| 耳机 | 171 | 154 / 462 | 17 / 51 | 513 |
| 键盘 | 154 | 139 / 417 | 15 / 45 | 462 |
| 鼠标 | 178 | 160 / 480 | 18 / 54 | 534 |
| 移动电源 | 172 | 155 / 465 | 17 / 51 | 516 |
| 保温杯 | 184 | 166 / 498 | 18 / 54 | 552 |
| 收纳箱 | 149 | 134 / 402 | 15 / 45 | 447 |
| 拖把 | 156 | 140 / 420 | 16 / 48 | 468 |
| 垃圾桶 | 176 | 158 / 474 | 18 / 54 | 528 |
| 合计 | 1,340 | 1,206 / 3,618 | 134 / 402 | 4,020 |

证据：[census](../reports/data/census/label_counts.csv)、[图片处理汇总](../data/processed/week1_v3/multimodal/summary.json)、[训练去重](../reports/data/training_deduplication_week2_v1.json)、[P5最终manifest](../reports/generation/lora/lora_instruction_data_v1_final_manifest.json)。

## 3. 字段定义与标准化

统一规则：src/data/pv_parser.py做Unicode NFKC和空白压缩；pv以#;#分项、#:#分键值，保存dict[str,list[str]]，值按原顺序去重。不做无来源缺失补全，不改数值/单位/区间原义，不把品牌或品类猜成型号。

| 字段 | 含义 / 标准化 | 核心属性与特殊值 |
|---|---|---|
| product_id | 商品数字字符串；长ID始终按文本 | 身份索引，非核心计分字段；Excel不得科学计数法 |
| raw_label / category_l1 / category_l2 | 来源标签 / 标准一级 / 二级；configs/data.json白名单映射 | 品类作为模型输入；不作为属性命中分母；reviewed=false不冒充全量人工确认 |
| title / raw.title | 标准化标题 / 原标题 | LoRA仅target排序、风格参考、审查；不进入生成输入。检索商品文本可用标题 |
| pv / raw.pv / attributes | 原属性串 / 解析多值字典 | 保留源证据；营销词和placeholder在下游门控，不全量改源 |
| description | 原生短描述，当前均为空 | 检索跳过；不以模型文案回填 |
| image_url / image_path / image_status / image_sha256 | URL / 相对路径 / 校验状态 / 图片指纹 | 检索输入；不是生成事实来源，只作源审计证据 |
| 品牌、型号 | 制造/标识与商品型号 | 多品类核心身份；无/否/无品牌不作正身份；型号=品类、纯容量等信息不足值按P5规则剔除 |
| 材质、外壳材质、杆材质、拖布材质 | 材料或部件材料 | 对应品类核心；“其他/见描述”等不推成具体材料，不推效果 |
| 容量、电池容量 | 容器体积或电池容量；单位原样保留 | 对应核心；必须保留数值/范围边界，多SKU不自行取一值；不把mAh当功率 |
| 保温时长、最大输出功率 | 时间区间、输出功率 | 对应核心；含/不含、上下界必须保持；不估算或取上界冒充定值 |
| 接口类型、充电协议 | 接口/协议枚举 | 键盘/鼠标/移动电源对应核心；不由USB推断无线/有线，不扩大兼容范围 |
| 耳机与播放设备连接方式、佩戴方式、耳机类别、蓝牙版本、功能 | 连接/佩戴/类型/版本/功能 | 耳机核心；连接层级不同可能并存或歧义，需独立审计 |
| 是否无线、是否机械键盘、是否有多媒体功能键、有无手托 | 布尔/枚举状态 | 键盘核心；“无/否/有线”可能是可靠否定事实，不等于placeholder |
| 工作方式、光学分辨率、按键数、无线技术、电源方式 | 鼠标原理/分辨率/按键/连接/供电 | 鼠标核心，部分亦为拖把核心；分辨率等需有数字，不能估值 |
| 电池类型、电芯类型 | 电池/电芯结构 | 移动电源核心，保留原枚举，不推出安全/耐用 |
| 尺寸、净重 | 规格/质量 | 收纳箱核心；0kg按P5占位处理，尺寸多值暂缓；不统一编造长宽高 |
| 杯子样式、形状、垃圾桶类型、开合方式 | 形态或机构枚举 | 对应核心；同义事实可语义命中不同源字段，不要求重复标签 |
| 拖把杆类型、挤水方式、驱动类型 | 拖把机构/脱水/驱动方式 | 拖把核心；可靠方式值不自动支持“高效/省力” |
| 适用场景、适用人群、适用对象、适用空间 | 源声明的范围 | 对应品类核心；允许来源明确集合，但不能扩展对象/场景或将类别猜成家庭/办公 |
| generation_input / used_attributes | 原推理输入 / 质量门控后实际可靠输入 | 与evaluation_attributes、core_attribute_count区分，不改正式分母 |
| task_type / target / cleaning_actions / template_version / quality_status | 三任务、监督文本、清洗动作、模板版本和质量来源 | P5逐条保存来源标题/属性/使用属性与版本；状态不是外部真值核验 |
| fact_id / fact_role / mandatory_status / evidence / source_sha256 | RAG事实身份/区域/门控角色/证据/hash | fact_id用于追踪，不把静态字段机会当真实事实数 |

冻结核心清单来自 configs/generation_evaluation.json：

- 耳机：品牌、型号、佩戴方式、耳机类别、耳机与播放设备连接方式、蓝牙版本、功能。
- 键盘：品牌、型号、接口类型、是否无线、是否机械键盘、是否有多媒体功能键、有无手托。
- 鼠标：品牌、型号、接口类型、工作方式、光学分辨率、按键数、无线技术、电源方式。
- 移动电源：品牌、型号、电池容量、电池类型、电芯类型、外壳材质、充电协议、最大输出功率。
- 保温杯：品牌、型号、材质、容量、保温时长、杯子样式、适用人群、适用场景。
- 收纳箱：品牌、型号、材质、尺寸、适用对象、适用空间、净重。
- 拖把：品牌、型号、杆材质、拖布材质、拖把杆类型、挤水方式、驱动类型、电源方式。
- 垃圾桶：品牌、外壳材质、形状、容量、垃圾桶类型、开合方式、适用场景。

## 4. 实际清洗流程与剔除规则

### A. 原始 → week1_v3（商品级）

src/data/preprocess.py / configs/data.json：

1. 流式扫描到SQLite候选库。标题/URL/pv必须为字符串；标题非空且≤100字符，pv≤12,000字符；图片HTTP/HTTPS域名必须在白名单；pv有解析issue或有效键<3则剔除。**未实现“核心字段缺失30%”这个PRD建议阈值，不能写成已执行规则**。
2. 白名单标签映射；标题排除配件、套装、工业等用途，具体各类词表在config。键盘用“笔记本键盘+维修/内置/适配标记”或PS/2组合规则排除内置替换件，不因“笔记本”一词删普通外接键盘。
3. ID、规范标题+排序属性的SHA、HTTP/HTTPS合并后的图片URL指纹唯一；重复内容不保留多份。
4. seed42+ID哈希每类选250；按类别内哈希80/10/余数分配，再做图片过滤，不重新分层补足。
5. src/data/images.py：实际格式JPEG/PNG/WEBP/GIF/BMP；文件≤12MiB，边长≥64，像素≤24,000,000；verify后再完整load，失败不造假图片；完全相同图片SHA只保留一件。4件失败+4件重复，2,000→1,992。

历史剔除：属性99、配件/套装863、用途56、替换键盘362、重复内容86、过长pv1，共1,467件；29,906→28,439。未命中计数的规则不写成历史实际剔除数。

### B. week1 train → week2_train_v1（只删新训练侧）

src/data/deduplication.py / configs/training_deduplication.json：规范品牌-型号相同，或同二级品类64位图片dHash汉明距离≤4，与冻结validation/test匹配时，从新train排除。1,592→1,559，排除33件；不移动或改写原split。同型号/近图是保守风险信号，不宣称全是同一SKU。

### C. P5 LoRA（商品级+字段级）

src/generation/lora_data.py / configs/lora_instruction_data_v1.json：

- 已确认CONFLICT排除15件，REVIEW暂缓28件。训练侧策略比RAG字段级REVIEW更保守，不把两者混写。
- 字段级删除：单值或全部组成词均占位的组合值、无效身份、型号=品类/纯容量/通用、数字字段不含数字、促销/物流营销值；NFKC/casefold去重。
- 非集合字段有多个不同值则整字段暂缓；功能、人群、场景、对象、空间、协议、耳机类别仅允许来源明确集合，至多3值；超过上限整字段暂缓。
- 否/无/未配备等可靠值可保留为输入约束，但不计可主动卖点；要求≥3个active可靠核心字段，不足剔除。
- 原标题必须含支持的品类别名；遇配件/组件边界或移动电源多容量冲突则剔除。title target只重组可靠组件并按原标题顺序，discard不可信自由文本，最多60字符。
- 卖点3条中性字段值；短详情最多5属性自然串联。不得由材质/品牌/品类推出安全、性能、场景等。
- 最终signature为SHA256([category_l2, title_target, reliable_attributes]的排序JSON)，非仅属性或ID。重复去31件，再按商品分层划分；所有任务同组。
- 排除/暂缓共219件：CONFLICT15、REVIEW28、可靠核心不足50、标题品类不支持56、配件边界39、final signature重复31，得到1,340件。
- 保存原标题、源属性、使用属性、清洗动作、三个target、来源ID、模板/数据版本；模型生成结果从不作训练真值。24件/72任务spot-check通过，只是抽查，不等于全量外部商品核实。

### D. RAG / 项目推理（不删除冻结评测商品）

src/generation/rag.py / rag_fact_policy_v3/v4：源missing不造事实；other/其他/未知/见描述/0kg等placeholder门控。REVIEW仅暂缓问题字段，普通CONFLICT仅阻断冲突字段；身份/品类冲突阻断整商品的事实输入，但保留评测商品与原核心分母。配件/服务边界通过源审计记录，不按输出好坏剔除test。

事实值规范化实际是NFKC+空白处理，不是完整物理单位换算器。numeric/范围矛盾由冻结policy审计及grounding规则处理；配置里的“单位比较”设计意图不能写成已对全量单位做语义转换。多值SKU不随意选一个；Identity/Mandatory/Supplemental/Negative区域见[技术说明](generation.md)。

源审计：project_validation 180PASS/11REVIEW/9CONFLICT；formal test100为84/11/5。PASS仅本轮未发现明显冲突，不代表真实商品页面全部参数验证。不会在人工模型评审中修改源status。

## 5. 三种分母、追溯与复现

validation静态 category×field opportunities=1,500；源存在非placeholder核心字段=1,159；Mandatory期望事实数与冻结core_attribute_count另行定义。静态missing不是注入失败，实际注入100%不等于输出命中100%。test100正式命中分母625不因门控/空输入改变。

[统一评测](evaluation.md)规定正确同义计命中、独立错误跨任务去重、源争议单列。原生标题不进入生成输入，但可以作检索文本及审计参考。

复现入口仅列定位，本轮不重跑：inspect_data.py、prepare_data.py、download_images.py、validate_data.py、deduplicate_training_data.py、build_lora_instruction_data.py。所有路径由configs管理；已有输出需要保留，重建需独立新目录并另行授权，尤其不得修改冻结validation/test。

数据/图片/任务JSONL作为附件，代码/config/小型统计可提交Git；P5 config/summary仍保留当时的p5_data_only或spot_check_pending字样，最终完成状态以P5 final_manifest为准，不能为“清晰”改冻文件。原始来源许可、全量源hash及未逐项核验字段需要人工确认。
