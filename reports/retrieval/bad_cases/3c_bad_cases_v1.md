# 3C 检索 Bad Case 分析（20 条）

- 版本：`week2_retrieval_3c_bad_cases_v1`
- 选择口径：从正式 test50 的 3C grade=0 结果中，每个二级品类5条、text/image各10条，优先高排名并覆盖不同根因；仅用于一次定性诊断，不用于权重选择。
- 数据边界：允许用冻结 test50 做一次定性根因分析；本报告不用于选择候选池、规则或权重。
- 公平性：Image→Text 的标准品类和属性只用于离线归因，不进入正式规则重排。

## 根因统计

| 根因 | 数量 | 比例 |
| --- | ---: | ---: |
| `cross_modal_alignment_insufficient` | 3 | 15% |
| `appearance_homogeneity` | 6 | 30% |
| `category_mismatch` | 1 | 5% |
| `attribute_conflict_or_data_noise` | 2 | 10% |
| `ranking_rule_insufficient` | 8 | 40% |

## 规则可修复性

| 判断 | 数量 |
| --- | ---: |
| `likely_fixable_by_parsed_text_category` | 1 |
| `likely_if_relevant_candidate_in_top20` | 8 |
| `not_fixable_without_image_signals` | 9 |
| `not_reliably_fixable` | 1 |
| `partly_fixable_by_quality_gate` | 1 |

## 逐例记录

| ID | 模式 | 品类 | 查询 | 原排名 | 错误候选 | 主因 | 规则可修复性 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| BC01 | text | 耳机 | 适合运动的无线蓝牙入耳式耳机 | 1 | 忆动2米台式电脑语音K歌吃鸡运动入耳式耳机带麦重低音直播耳塞 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC02 | text | 耳机 | 带麦克风和线控的2米有线入耳式耳机 | 2 | gh原装入耳式有线耳机适用OPPOA11Reno4A92s8Ace2K5FindX2A52通用 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC03 | text | 耳机 | 3米线长有线入耳式监听耳机 | 2 | 忆动2米台式电脑语音K歌吃鸡运动入耳式耳机带麦重低音直播耳塞 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC04 | image | 耳机 | 带麦克风和线控的2米有线入耳式耳机 | 1 | DACOM Athlete+洗澡运动蓝牙耳机骑行重低大音量跑步音乐听歌耳塞 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC05 | image | 耳机 | 蓝牙挂脖式磁吸运动耳机 | 1 | 美国Qbuds乔耳小方盒蓝牙无线耳机双耳运动适用苹果小米半入耳 | `cross_modal_alignment_insufficient` | `not_fixable_without_image_signals` |
| BC06 | text | 键盘 | 带RGB灯效和手托的有线机械键盘 | 1 | 雷蛇技术正品新盟牧马人真机械键盘鼠标套装青轴黑轴吃鸡lol外设 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC07 | text | 键盘 | 兼顾办公和游戏的有线机械键盘 | 3 | 双飞燕 X7-G800V QQ炫舞游戏专业键盘有线USB劲舞团打P吃鸡宏编程 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC08 | image | 键盘 | 带RGB灯效和手托的有线机械键盘 | 1 | 黑轴有线游戏机械键盘台式电脑usb背光发光办公用金属跑马灯防水 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC09 | image | 键盘 | 有线办公键盘 | 2 | Rii i8迷你无线小键盘 家用USB充电键鼠电视机顶盒HTPC笔记本电脑 | `cross_modal_alignment_insufficient` | `not_fixable_without_image_signals` |
| BC10 | image | 键盘 | 带多媒体功能键的有线机械键盘 | 2 | 卡通可爱女生少女心套装鼠标键盘USB有线电脑台式笔记本家用粉色 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC11 | text | 鼠标 | 无线静音游戏鼠标 | 1 | 。我要买机械鼠标通用Usb插口的玩穿越火线程序员网吧无声发光的 | `attribute_conflict_or_data_noise` | `partly_fixable_by_quality_gate` |
| BC12 | text | 鼠标 | 1600DPI有线游戏鼠标 | 1 | 楚王158鼠标有线USB家用办公商务PS2圆口台式笔记本电脑长线1.8米 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC13 | image | 鼠标 | 1600DPI有线游戏鼠标 | 1 | 楚王158鼠标有线USB家用办公商务PS2圆口台式笔记本电脑长线1.8米 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC14 | image | 鼠标 | 1600DPI有线游戏鼠标 | 3 | 爱国者Q710 无线2.4G鼠标 台式机笔记本专用无线鼠标办公精品鼠标 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC15 | text | 鼠标 | 电竞游戏鼠标 | 8 | 黑轴有线游戏机械键盘台式电脑usb背光发光办公用金属跑马灯防水 | `category_mismatch` | `likely_fixable_by_parsed_text_category` |
| BC16 | text | 移动电源 | 适用于苹果手机的5000mAh超薄自带线充电宝 | 1 | 双向快充无线充电宝10000毫安苹果华为vivo通用oppo移动电源便携 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC17 | text | 移动电源 | 便携的10000mAh充电宝 | 2 | ROMOSS/罗马仕 30000m毫安充电宝大容量8p 手机电充宝快充 闪充移动电源旗舰店官方旗舰七千猫舰旗超大量式士 | `ranking_rule_insufficient` | `likely_if_relevant_candidate_in_top20` |
| BC18 | image | 移动电源 | 自带线和充电插头的三合一充电宝 | 4 | 双向快充无线充电宝10000毫安苹果华为vivo通用oppo移动电源便携 | `appearance_homogeneity` | `not_fixable_without_image_signals` |
| BC19 | image | 移动电源 | 适用于苹果手机的5000mAh超薄自带线充电宝 | 1 | 可爱女生卡通超萌苹果vivo华为oppo手机通用充电宝迷你毫安大容量移动电源超薄小巧便携快充创意大容量 | `attribute_conflict_or_data_noise` | `not_reliably_fixable` |
| BC20 | image | 移动电源 | 10000mAh快充充电宝 | 1 | Eary一粒充电伴侣自带线充电宝充电器三合一便携小巧迷你大容量 | `cross_modal_alignment_insufficient` | `not_fixable_without_image_signals` |

## 逐例依据

- **BC01：** 查询要求无线蓝牙入耳式，候选属性明确为有线；文本属性冲突可由规则降权。
- **BC02：** 候选虽为入耳式，但缺少2米线长和带麦克风等主要约束；细粒度属性未进入原始排序。
- **BC03：** 查询明确3米监听耳机，候选为2米游戏/线控耳机；数字和用途约束可解释降权。
- **BC04：** 两者均呈现耳机主体，视觉外形相近，但查询商品为有线入耳式，候选为蓝牙挂耳式；正式图片查询没有属性信号。
- **BC05：** 查询图是挂脖磁吸耳机，候选图是充电盒式真无线耳机，形态差异明显但文本候选仍排第一，说明细粒度图文形态没有对齐。
- **BC06：** 查询要求RGB灯效和手托，候选标题只稳定支持机械键盘/鼠标套装，关键功能匹配不足。
- **BC07：** 查询要求兼顾办公和游戏的机械键盘，候选为普通有线游戏键盘，机械属性不足。
- **BC08：** 均为深色机械键盘主体，候选黑轴键盘与查询RGB/手托细节不一致，图像全局特征更关注共同轮廓。
- **BC09：** 查询是有线办公键盘，候选是无线迷你小键盘；文本商品语义与查询图片的连接方式/尺寸没有对齐。
- **BC10：** 两者都包含键盘外观，候选偏卡通键鼠套装且缺少机械/多媒体约束；无图片侧属性解析时无法规则纠正。
- **BC11：** 候选标题含“我要买”等查询式噪声，且USB机械鼠标描述与无线静音要求冲突；标题污染会误导文本匹配。
- **BC12：** 查询要求1600DPI有线游戏鼠标，候选仅支持普通有线办公/商务，缺少DPI和游戏属性。
- **BC13：** 同为常见鼠标轮廓，候选办公鼠标在视觉上接近游戏鼠标，但关键DPI/用途无法从当前规则获得。
- **BC14：** 候选是无线2.4G鼠标，与有线游戏鼠标外观接近；连接方式冲突未被现有跨模态表示充分区分。
- **BC15：** 电竞游戏鼠标文本查询召回键盘商品；文本中可直接解析鼠标品类，可用二级品类系数降权。
- **BC16：** 查询要求5000mAh超薄自带线，候选为10000mAh无线/双向快充，容量和功能均冲突。
- **BC17：** 查询要求便携10000mAh，候选为30000mAh大容量；数值约束可直接用于降权。
- **BC18：** 查询商品为自带线和插头的三合一移动电源，候选为无线/双向快充移动电源；同类外观掩盖了结构功能差异。
- **BC19：** 候选标题容量信息缺失且历史审计显示容量声明可疑；图片外观相似与源属性质量问题同时存在。
- **BC20：** 查询图是黑色18W快充移动电源，候选图是彩色自带线/插头三合一产品，外观和功能呈现均有差异，现有图片到文本对齐仍将其排第一。
## 结论

- 文本查询中的明确品类、连接方式、规格数字和功能冲突，可由更深候选池上的规则重排修正，前提是相关商品已进入候选池。
- 外观同质化、图片侧细粒度属性缺失和跨模态对齐不足，无法由当前正式 Image→Text 规则修正。
- 标题污染、属性错位或容量声明异常属于源数据质量问题，不应靠提高规则权重掩盖。
- 跨品类误召回在文本查询可由解析出的品类降权；图片查询缺少独立品类识别器时不能使用隐藏 metadata 修正。
