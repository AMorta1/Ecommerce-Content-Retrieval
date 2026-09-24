# Week 2 P1-RAG validation / test100 只读事实质量审计

审计版本：`week2_rag_fact_quality_audit_v1`  
validation 数据：`data/processed/week1_v3/multimodal/validation.jsonl`，SHA256 `dd2bc1d6c21567176d95f5414bfd9304f48979b3bb39ab0b038627b1f346af0b`  
test100 数据：`data/processed/week1_v3/generation_evaluation_samples.jsonl`，SHA256 `ce057356a9a0c48e4298daac545fafb0c83ad07ae2f74f1b1211c834b048a557`

## 范围与方法

- 全量检查 validation 200 件与固定 test100 100 件，共 300 件。
- 对全部 300 件扫描标题、结构化属性及字段内一致性，并逐张查看 300 张本地主图。
- 重点检查品类/商品身份、品牌型号、容量规格、材质、连接方式、机械/非机械、配件/整机及多 SKU 混写。
- 原始数据、图片、冻结样本和人工标签均未修改、删除、替换或重抽。
- `PASS` 只表示本轮未发现明显冲突，不等于已经从原商品页确认所有事实。
- test100 结果只用于后续错误归因，不参与字段白名单、Top-K、阈值或 Prompt 选择。

## 统计

| 范围 | 样本数 | PASS | REVIEW | CONFLICT |
| --- | ---: | ---: | ---: | ---: |
| validation | 200 | 180 | 11 | 9 |
| test100 | 100 | 84 | 11 | 5 |
| 合计 | 300 | 264 | 22 | 14 |

逐条结果见 `reports/data/week2_rag_fact_quality_audit.csv`。

## 已确认冲突

| 范围 | 商品 ID | 品类 | 问题类型 | 依据 |
| --- | --- | --- | --- | --- |
| validation | 565078231259 | 垃圾桶 | brand_title_attribute_conflict | 标题和主图品牌为 ZITALEN，结构化“品牌”为 1208S，疑似把型号写入品牌字段。 |
| validation | 580298584089 | 收纳箱 | material_title_image_attribute_conflict | 标题和主图均为布艺收纳箱，结构化“材质”为塑料。 |
| validation | 587823728552 | 耳机 | product_identity_conflict | 标题指向 VIVO/iQOO 原装耳机，主图与结构化品牌/型号指向 PLEXTONE G15。 |
| validation | 600255465931 | 耳机 | wireless_title_image_attribute_conflict | 标题写无线蓝牙，结构化“是否无线”为有线；主图可见连接线。 |
| validation | 601943728885 | 保温杯 | capacity_title_attribute_conflict | 标题写 2200mL，结构化“容量”为 1200mL。 |
| validation | 610888219389 | 移动电源 | capacity_title_image_attribute_conflict | 标题和主图宣称 1000000M，结构化电池容量为 20000mAh，型号也写成 100000M。 |
| validation | 616914231244 | 移动电源 | capacity_title_image_attribute_conflict | 标题和主图宣称 1000000M，结构化电池容量为 20000mAh。 |
| validation | 617219746831 | 移动电源 | capacity_values_conflict | 标题同时出现 1000000 和 20000mAh，结构化电池容量为 20000mAh，无法确定所选 SKU 的可信容量。 |
| validation | 624000064339 | 拖把 | brand_title_attribute_conflict | 标题品牌为澳美森，结构化品牌为蜀丽康，主图不能判定哪一方正确。 |
| test100 | 583014299804 | 移动电源 | model_attribute_conflict | 标题、主图和其他属性均为移动电源，结构化“型号”却为“键盘”。 |
| test100 | 589508217209 | 移动电源 | capacity_title_image_attribute_conflict | 标题包含 1000000/80000 容量宣称、主图写 1000000M，结构化电池容量为 3000mAh。 |
| test100 | 566531828259 | 保温杯 | capacity_attributes_conflict | 结构化杯子容量值为 450mL，但容量区间为 301–400mL。 |
| test100 | 609629279071 | 保温杯 | capacity_and_material_conflict | 标题和主图为双层玻璃杯；结构化材质为304不锈钢，杯子容量值520mL却对应301–400mL区间。 |
| test100 | 599485472276 | 收纳箱 | capacity_size_image_conflict | 标题和主图为可承重的大型塑料周转筐，结构化容量为2L且尺寸值异常，明显不相称。 |

## 待复核

| 范围 | 商品 ID | 品类 | 问题类型 | 依据 |
| --- | --- | --- | --- | --- |
| validation | 41137975381 | 移动电源 | accessory_boundary | 商品是免焊接移动电源盒/充电宝壳，电池容量为 0mAh；需确认配件是否应纳入移动电源生成范围。 |
| validation | 526413502198 | 鼠标 | service_listing_boundary | 标题主要销售鼠标维修/更换微动服务，主图是鼠标；需确认生成对象是服务还是商品。 |
| validation | 552025795571 | 收纳箱 | category_boundary | 标题和主图为医疗废物周转箱，当前二级品类为收纳箱；需确认医疗废物箱是否纳入该品类。 |
| validation | 579600375674 | 耳机 | sku_component_boundary | 标题包含 AirPods 单只补配，结构化属性未说明单只/整套；需确认具体 SKU。 |
| validation | 586203674883 | 收纳箱 | capacity_plausibility | 标题和主图为多尺寸木质收纳箱，结构化容量仅 1L；需结合具体尺寸/SKU 核实。 |
| validation | 590454993449 | 耳机 | connection_variant_ambiguity | 标题同时出现有线、蓝牙和 DIY，结构化属性仅为有线；主图只能确认有线耳机。 |
| validation | 597102621828 | 保温杯 | capacity_variant_ambiguity | 标题出现 1000mL、1L、800mL，主图出现 900/750mL，结构化值为 900mL；存在多规格混写。 |
| validation | 597328189786 | 鼠标 | connection_variant_ambiguity | 标题同时包含有线、无线、蓝牙，结构化属性指向蓝牙，主图未给出明确连接方式。 |
| validation | 606828457125 | 移动电源 | implausible_capacity_claim | 标题和结构化属性均写 1000000mAh，但体积与常见产品明显不相称；缺少外部规格证据。 |
| validation | 609973546605 | 移动电源 | station_product_boundary | 商品是共享充电宝机柜，标题把 5000mAh 误写为 5000毫升；需区分机柜与柜内单个电源。 |
| validation | 614287670938 | 耳机 | component_boundary | 标题和主图更接近耳机单元/喇叭组件，需确认是否属于完整耳机商品。 |
| test100 | 522783084872 | 移动电源 | capacity_claim_plausibility | 结构化容量为5400mAh，型号含20000M，主图宣称可充手机10次；需外部规格核实。 |
| test100 | 621846828786 | 移动电源 | station_product_boundary | 商品为14口共享充电机柜，结构化容量可能对应柜内单个电源，需区分机柜与电源。 |
| test100 | 600661711292 | 耳机 | image_not_decisive | 主图只展示联名包装盒，未清楚展示耳机本体，无法用图片核实标题和属性。 |
| test100 | 2859211238 | 鼠标 | connection_variant_ambiguity | 标题同时出现G502 HERO有线与G502无线，主图和结构化型号偏向无线版，需确认具体SKU。 |
| test100 | 544826941132 | 鼠标 | service_listing_boundary | 标题主要销售鼠标维修/更换微动服务，主图为鼠标；需确认生成对象。 |
| test100 | 545593037598 | 保温杯 | capacity_variant_ambiguity | 杯子容量值同时含260mL和320mL，而容量区间为301–400mL；需确认具体SKU。 |
| test100 | 591309078741 | 保温杯 | capacity_variant_ambiguity | 杯子容量值同时含340mL和450mL，而容量区间为301–400mL；需确认具体SKU。 |
| test100 | 609762128375 | 保温杯 | capacity_variant_ambiguity | 杯子容量值同时含280mL和500mL，而容量区间为201–300mL；需确认具体SKU。 |
| test100 | 583457250405 | 拖把 | bundle_component_boundary | 标题强调通用拖把杆，主图主要展示甩水篮组件，结构化属性又像整套拖把；需确认SKU。 |
| test100 | 621173570641 | 拖把 | bundle_component_boundary | 标题强调通用拖把杆，主图为整套桶和拖把，结构化品牌/型号均为“见描述”；需确认SKU。 |
| test100 | 624392441174 | 收纳箱 | category_boundary | 标题和主图更接近叠衣板/抽屉整理架，当前二级品类为收纳箱；需确认品类边界。 |

## 主要数据问题

1. **容量字段风险最高**：移动电源存在 1000000M 与 20000/3000mAh 冲突，保温杯存在单值与容量区间冲突，且多 SKU 容量常被合并在同一记录中。
2. **商品身份和品类边界**：移动电源壳、共享充电机柜、鼠标维修服务、耳机单元、拖把杆/组件等记录不能直接按完整商品生成。
3. **品牌/型号错位**：少数记录把型号写入品牌，或标题品牌与结构化品牌完全不同。
4. **材质与连接方式冲突**：存在布艺箱写成塑料、无线耳机写成有线等明确源事实错误。
5. **多值属性不是默认可用事实**：颜色分类、套餐、容量、尺寸和连接方式中混有不同 SKU；不能把所有值同时注入 Prompt。
6. **图片证据有限**：部分主图只展示包装、场景或组件，只能标记 REVIEW，不能据此判断结构化事实正确。

## 事实知识单元设计

机器可读设计见 `configs/rag_fact_policy_v1.json`。每个事实单元至少记录：

- `fact_id`、商品 ID、数据集版本、split；
- 一级/二级品类；
- PRD 四级中的 `category / function / parameter / specification`；
- 原字段、规范字段、原值和规范值；
- 适用任务：标题、卖点、短详情；
- 质量状态：`eligible / withhold_review / blocked_conflict`；
- 标题/图片审计证据及源文件哈希。

事实只来自同一商品的标准品类和结构化属性。原标题和主图只作为质量审计证据，不作为首版 RAG 注入内容；模型生成文案、Baseline 输出、人工评价和其他商品均不能成为事实来源。

## validation 初始字段建议

### 建议纳入

- 全品类身份信息：标准品类、经审计可用的品牌和型号。
- 耳机：连接方式、是否无线、蓝牙版本、佩戴方式、耳机类别、单双耳。
- 键盘：接口、是否无线、是否机械、轴体、键数、多媒体键和手托。
- 鼠标：接口、无线技术、工作方式、DPI、按键数和电源方式。
- 移动电源：容量、充电协议、输出功率、电池/电芯、材质和尺寸；容量必须先通过数值与 SKU 一致性门槛。
- 保温杯：材质、容量、保温时长、杯型和吸管信息；容量单值与区间冲突时禁用。
- 收纳箱：材质、尺寸、容量。
- 拖把：杆/拖布材质、挤水方式、驱动类型、拖把类型和桶尺寸。
- 垃圾桶：材质、形状、容量、类型和开合方式。

### 建议排除或条件使用

- 排除颜色分类、套餐、促销、价格、销量、售后、保修、产地、生产企业、出口资质、包装体积等字段。
- 收纳箱的适用对象/空间和其他大量枚举场景字段噪声较高，首版排除。
- “快速充电、强力吸水、长效保温、降噪、防水”等效果性字段只有在规范字段明确且无冲突时条件使用，不能从标题营销词推导。
- 多值字段只有在值可同时成立时保留；不同 SKU 的规格混写一律暂缓。
- REVIEW 记录只暂缓问题字段；CONFLICT 记录阻断冲突字段。若品类或商品身份冲突，则该商品全部事实不得进入生成上下文。

## 初始 Top-K 建议

该建议只依据 validation、PRD 和 Week2 约束：

- 商品必要身份信息单独提供，不计入 Top-K；
- 首版最多召回 **Top-3** 个 `eligible` 任务相关事实；
- 候选只来自同一商品 ID；
- 按任务字段优先级和固定字段顺序确定，暂不提出数值相似度阈值；
- 不足 3 条时返回更少事实，不用低置信度事实或其他商品事实补齐。

本阶段未加载 Qwen、未执行 Prompt 或 RAG 推理，因此没有新的模型 revision/snapshot 需要记录。下一阶段如实际加载模型，必须记录 resolved revision 和本地 snapshot。
