# P7：全量配对生成完成，固定32件人审待填写

## 当前结果与停止点

Base和LoRA各完成原project_validation全200件、600个独立任务（合计1200次）。两侧只有adapter加载差异，600对逐商品/任务seed、Prompt hash与token/preflight一致，无参数更新，无空输出，无输出达到320 token上限，模型参数未变化。离线固定模型revision，未读取project_test、未运行或读取RAG holdout归档。没有commit/push、没有修改P5或训练配置。

用户最新要求改为分层32件人审，不做200件全量人工复核。8类各4件、seed42；只从源ID/品类SHA256排序选择，不使用输出、质量状态或人工结果。A/B每类2/2，seed41008，独立匿名顺序。抽样是在Base生成开始后、人工质量结果前固定，**不是generation-output holdout**，也不是RAG holdout。冻结之后统计恰好32件均为源审计PASS，非按PASS筛选；不重抽。未覆盖全200中的11个REVIEW和9个CONFLICT，因此人审结论不能证明这些源争议商品同样安全。

工作簿：`quick32_v1/project_validation32_blind_review.xlsx`，32行商品/64份文案。人工三指标、四项匿名偏好、机械模板化等诊断均未预填；映射veryHidden，源属性/核心清单/可靠输入/A/B三任务同一行。正式人工分母仅样本对应186个核心属性，不把样本标签冒充全200的1167个核心属性标签。

**尚无正式人工质量结果，不能判断LoRA已经符合晋级条件。** 当前状态为自动生成完成、人审待填写。填写完整32件后先匿名汇总、后解盲，才能讨论事实性/覆盖是否明显退化与表达是否有收益；用户确认后再决定是否进入test，本轮不执行test。

## 全200件自动诊断（不是人工指标）

| 诊断 | Base | LoRA |
| --- | ---: | ---: |
| 字面核心字段命中 | 986/1167（84.49%） | 948/1167（81.23%） |
| 三任务结构均符合冻结解析规则 | 94/200 | 197/200 |
| 命中评价/效果疑似词的商品 | 159/200 | 3/200 |
| 疑似未字面支持的数值单位商品 | 5/200 | 0/200 |
| 空输出任务 / 达到token上限任务 | 0/600、0/600 | 0/600、0/600 |

字面覆盖下降约3.26个百分点需要人审确认是真遗漏还是正常同义/格式；不能直接认定正式属性覆盖退化。疑似词下降不能替代事实错误率，词可能源支持或属于误报。结构解析接受三项JSON数组或三条编号/bullet；非该格式也完整保留原文，不等于文案不通顺。所有判断都包含失败解析输出，不挑成功子集。

## 本轮三调用延迟（秒，包含首个cold call）

| 版本 | 标题平均 | 卖点平均 | 短详情平均 | 三任务总平均 | 三任务总p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1.890 | 3.131 | 3.120 | 8.141 | 31.483 |
| LoRA | 1.211 | 1.605 | 2.203 | 5.019 | 8.213 |

Base进程wall 1645.21秒（27.42分钟），LoRA1020.42秒（17.01分钟）。峰值allocated分别5497.7/5507.4MiB，reserved5628/5636MiB。延迟从generate至GPU同步完成，预处理/tokenization/解码及模型加载不计入单调用延迟，wall另报；总调用时间包含输出长度和包装开销，不归因对比旧一次调用Baseline。逐商品完整延迟在 `latency_by_product.csv`。

## 冻结版本/hash链

- 固定基座revision：`a09a35458c702b33eeacc393d103063234e8bc28`，本地snapshot：`E:\Projects\Baidu\.model-cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28`。
- epoch3/step1359 adapter权重：`890a061812321ccf6bbf598b37617fc59dcde70d57042c4304c219a0f6740e31`。
- project_validation：`dd2bc1d6c21567176d95f5414bfd9304f48979b3bb39ab0b038627b1f346af0b`。
- 任务Prompt沿用P5 `lora_copy_templates_v1`；chat-template：`2e2d2512cfe46af53dc1eed45368ecaab26ac4461c480e6b69fe571cf0ceaa75`。
- 推理配置 `configs/lora_project_validation_v1.json`：`4a86013f8e0bae471c6e75694d730232c580465cbc01df630ee8640636d3b1c5`。
- 首次生成前 `protocol_manifest.json`：`095cb45a3c07b86ccace377d746ffe3224a9805aaf5204f19640652e08aa1ab2`，含Git HEAD `08d2e8fdadd3540495563edd2ad3fa5d33e839b4`、未提交状态、代码原始字节快照、数据/配置/token/环境hash，不将未提交代码等同于HEAD。
- 新快速人审配置：`94dc6c2fab251759833139181103d363742cf794715aee63f9c35df033428772`；`quick32_v1/sampling_manifest.json`：`d77a9462635099c907c116aa80ddce94cc5fada4971fe34074fb79d12ea1ac9d`。
- 全量结果与初始空白工作簿归档 `automatic_phase_manifest.json`：`d9de118a82768bb07cd362f783e141565aa961d498fb80543734a79fb21ffae4`；初始xlsx：`5c23171b28c9944e517d607453918ceaad96ce7d8591916798293979cc84971e`。人工填写后xlsx hash应变化；冻结源/文案/映射不应变化。

原冻结推理协议中的200件人审计划不覆写，新 `sampling_manifest.json` / `docs/lora_project_validation_quick32_v1.md` 按最新用户授权仅替代人审范围；不改变任何推理设置和输出。代码/配置/数据/adapter漂移会报错。环境为Python3.10.21、Torch2.12.0+cu130、Transformers5.16.1、PEFT0.21.1、bitsandbytes0.50.2、accelerate1.14.0，不安装或升级包。

## 你接下来需要做

1. 打开32件xlsx并读“复核说明”，只填黄色列。
2. 对A/B各填matched_attribute_count、fluency_pass、factual_error_count；四项偏好填A更好/B更好/持平；请填写两侧mechanical_template 0/1，其他诊断也建议填。错误或诊断1须notes引用原文/依据。
3. 每行填reviewer与review_confirmed=1，允许分批保存，不改ID/源信息/分母/输出。不查看匿名映射或带版本名输出。
4. 全32件保存后告诉我，再核验并汇总。小样本仅为方向性证据，不自动设新数值硬门槛，不因自动疑似词较少就宣布事实问题解决。

未来完整人审经确认后的入口（现在不要运行）：

```powershell
Set-Location 'E:\Projects\Baidu\Ecommerce-Content-Retrieval'
& 'E:\Anaconda\python.exe' -X utf8 scripts/review_lora_project_validation_quick32.py --mode summarize
```

建议提交范围：新增P7两个配置、两份协议、推理/复核/归档脚本、`src/generation/lora_validation*.py`、两个P7测试，以及本轮reports（包括完整配对输出与清单；匿名映射评审前不要查看）。原P5/P6未提交文件另按其既有归档处理，不混作本轮新增。不要提交7B基座/大权重、Python缓存，不自动commit/push。
