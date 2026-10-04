# RAG v2 P2.5 development24 人工诊断（复核定稿）

## 范围与约束

- 只复核 `rag_v2_p2_development24.json` 中已有的 24 个商品、3 个配对版本，共 72 条输出。
- 未重新生成；未运行或读取 32 条 generation-output holdout；未修改 Prompt、policy、输出或 validator。
- 输入文件 SHA-256：`78accdb37ae70916559368d32ebff439ed8f35f7998cba3d43249787eb50a8c0`。
- 最终复核工作簿：`rag_v2_p25_development24_review_workbook.xlsx`；复核定稿版本 SHA-256：`55b61664955fce7a03a298dfca964e6abdb76e4971acbd191ef5d8965ff06be9`。
- 三个版本共 72 组复核项均已填写：69 组一致、3 组需修改、0 组不确定；本报告已采纳全部 3 处修改。
- 这是 development outputs 的诊断，不是 P3 或 formal test100 结果。

## 人工评测口径

- `matched_attribute_count`：在冻结的核心属性清单中，值在标题、三个卖点或短详情任一处得到正确表达即计 1；语义等价、单位大小写或不改变边界含义的格式改写计命中；表达错误不计命中。
- `fluency_pass`：标题、三个卖点和短详情作为整体均语法可读、表达自然则为 1，否则为 0。
- `factual_error_count`：统计相互独立的无依据或矛盾事实；同一错误跨多个位置重复只计 1。
- 核心属性命中率分母保持正式评测的冻结 `core_attribute_count`，24 条合计为 134；没有把 1500 个静态 field opportunities 或 Mandatory 注入条数替换成正式分母。
- 事实错误分类中的 `supported paraphrase / validator false positive` 仅用于诊断自动 Grounding，不计入 `factual_error_count`。

## 汇总

| 版本 | 核心属性命中 | 人工核心属性命中率 | 有事实错误样本 | 事实错误样本率 | 事实错误总数 | 平均事实错误数 | 通顺样本 | 通顺率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1-control | 105 / 134 | 78.36% | 11 / 24 | 45.83% | 17 | 0.71 | 22 / 24 | 91.67% |
| V2-context | 120 / 134 | 89.55% | 17 / 24 | 70.83% | 65 | 2.71 | 24 / 24 | 100.00% |
| V2-coverage | 119 / 134 | 88.81% | 17 / 24 | 70.83% | 57 | 2.38 | 24 / 24 | 100.00% |

人工结果说明：两个 V2 版本均显著提高了 development24 的核心属性覆盖和通顺率，但生成了大量没有被源数据支持的评价、效果或场景表达。这个结论来自人工逐条复核，而不是由自动 Grounding PASS 数直接推导。

## V2-context：70/76 字面覆盖中的 6 个未覆盖 Mandatory fact

| product_id | 字段 | Mandatory 原值 | 输出表达 | 人工判定 | 对正式指标的处理 |
|---|---|---|---|---|---|
| 576694559149 | 保温时长 | `6小时(含)-12小时(不含)` | `6小时-12小时` | 正常格式表达 | 属性命中；不计事实错误 |
| 576694559149 | 容量 | `401mL(含)-500mL(含)` | `401mL-500mL` | 正常格式表达 | 属性命中；不计事实错误 |
| 584648531319 | 容量 | `301mL(含)-400mL(含)` | `300mL` | 错误表达 | 不命中；计 1 个 provided fact contradiction/distortion |
| 600702277714 | 保温时长 | `6小时(含)-12小时(不含)` | 卖点为 `6小时-12小时`，标题及短详情又写 `4-12小时` | 错误表达 | 属性仍命中，但计 1 个 provided fact contradiction/distortion |
| 600702277714 | 容量 | `501mL(含)-600mL(含)` | `501mL-600mL` / `501毫升-600毫升` | 正常格式/单位表达 | 属性命中；不计事实错误 |
| 610408134586 | 电池容量 | `450000mah` | `450000mAh` | 正常单位大小写表达 | 属性命中；不计事实错误 |

结论：6 个字面未覆盖项中，4 个是正常同义/格式表达，2 个是错误表达，0 个是真遗漏。`70/76` 是字面诊断，不应直接当作正式人工属性命中率。

## 自动 Grounding FAIL 的人工原因拆分

下面的“涉及 FAIL 样本数”允许同一条样本落入多个原因，因此各行不能相加还原 18。`validator false positive` 指自动命中的具体 finding 有源支持；它并不保证该条输出完全没有其他人工事实错误。

| 人工原因 | V2-context 涉及 FAIL 样本数 | V2-coverage 涉及 FAIL 样本数 |
|---|---:|---:|
| unsupported generation | 3 | 3 |
| provided fact contradiction / distortion | 2 | 0 |
| evaluative claim | 14 | 12 |
| unsupported scenario | 5 | 5 |
| identity/category error | 0 | 0 |
| supported paraphrase / validator false positive | 8 | 9 |

进一步核对：

- 两个版本的自动 Grounding 均为 6 PASS / 18 FAIL。
- V2-context 的 18 条自动 FAIL 中，15 条人工确认至少有 1 个事实错误，3 条是纯误报；另有 2 条自动 PASS 被人工发现事实错误。
- V2-coverage 的 18 条自动 FAIL 中，13 条人工确认至少有 1 个事实错误，5 条是纯误报；另有 4 条自动 PASS 被人工发现事实错误。
- V2-coverage 的身份错误出现在自动 PASS 的商品 `580390189481`：输出无依据写入“微软”，所以 Grounding FAIL 子集中的 identity/category error 为 0，并不表示该版本整体没有此类错误。
- 常见误报来源是：原始标题已支持“大容量/便携/台式与笔记本通用”，结构化字段已支持“家庭使用”，以及“拖把用于家居清洁”“收纳箱用于家居收纳”这类直接品类语义。

## 逐条明细

逐条的三个正式人工字段、事实错误类别和判断依据见 `rag_v2_p25_development24_human_review.csv`。

## 最终复核修正

- `610727155009 / V2-coverage`：`factual_error_count` 从 0 改为 1；“确保有效挤出水分”计 1 个无依据评价性效果保证。“家居清洁”仍由源标题和拖把直接品类语义支持，因此自动场景 finding 仍记为 validator false positive。
- `621656965008 / V1-control`：`factual_error_count` 从 0 改为 1；将原始标题明确支持的安卓/苹果/华为/小米手机范围泛化为“各种3C设备”，计 1 个 unsupported scenario。
- `621656965008 / V2-context`：`factual_error_count` 从 3 改为 4；将明确的手机兼容范围泛化为“各种播放设备”，增加 1 个 unsupported scenario。
- 三处修正均未改变 `matched_attribute_count` 或 `fluency_pass`。

## 当前诊断结论

- 不能因为自动 Grounding PASS 从 12/24 降到 6/24 就直接认定 V2 事实错误率恶化；自动结果同时存在误报和漏报。
- 但按同一人工口径，当前两个 V2 在这 24 条 development outputs 上的事实错误确实高于 V1-control，主要增量是无依据评价性表达，其次是无依据场景。
- V2-context 与 V2-coverage 的人工核心属性命中率接近；V2-context 略高 1 个命中，但事实错误也更多。
- 本诊断只用于决定是否需要后续 P2 Prompt 版本，不构成对 P3 holdout 的结论。
