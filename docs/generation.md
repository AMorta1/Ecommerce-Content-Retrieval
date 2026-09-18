# 文案生成基线

## 当前范围

Week 1 使用 `Qwen/Qwen2.5-7B-Instruct` 验证中文电商文案生成。输入只包含标准化后的品类和商品属性；原始商品标题不进入模型，避免目标泄露。输出固定为一个生成标题、三个卖点和一段短描述。

Prompt 按 PRD 的“角色设定 + 任务说明 + 属性输入 + 输出要求”组织，并给三类内容分别设置规则。3C 数码强调型号、参数、连接方式和功能，家居日用强调材质、规格、场景和实用性。三类内容在一次推理中统一输出，减少重复加载和推理开销。

模型通过 Transformers 加载，并用 bitsandbytes 在运行时压缩为 NF4 4-bit，以适配本地 RTX 4070 Laptop 8GB 显存。原始模型权重只缓存在仓库上一级的 `.model-cache/huggingface/`，不提交 Git。

当前五条样本是人工检查过的链路验证样本，不是正式文案标注集。它们只能用于检查模型能否加载、中文输出是否可用、JSON 格式是否稳定，不能据此宣称模型达到正式效果指标。

## 可行性结果

2026-09-09 在本地 RTX 4070 Laptop 8GB 上完成五条离线试跑：五条输出都通过 JSON 结构校验，证明推理链路可用。

正式 Baseline 从 test 集分层随机抽取100条，两大类各50条，随机种子为42；8个细分类各12或13条。生成输入要求至少有4个核心属性，并按配置剔除常见单值占位符；组合写法的占位值仍有少量残留，并在人工备注中标出。最终96条严格满足 JSON 和三个卖点的格式，结构成功率为96%；平均生成时间6.443秒，P95为7.759秒，最慢8.887秒，PyTorch峰值显存约5817MiB。平均时间低于 PRD 的12秒上限。

自动字符串匹配发现470/625个核心属性值，比例为75.2%。这个数值会漏掉同义表达，也不能识别事实错误，因此只是标注辅助信息，不是 PRD 要求的正式人工相关度。

内容初查发现模型会把已有事实扩写成未提供结论，例如把“塑料”写成“环保塑料”、把普通键盘写成“游戏键盘”。这不影响“模型能否运行”的可行性结论，但说明当前结果还只是 Baseline。正式人工评测已经加入事实一致性指标，不能只检查语言是否流畅。

100条人工复核已经完成。核心属性命中550/625，命中率88%；通顺率99%；品类风格通过率100%。三项PRD基础门槛——属性命中率不低于75%、通顺率不低于80%、平均生成不超过12秒——均已通过。事实错误样本率为56%，平均每条1.05项，说明模型的主要短板不是语言表达，而是容易生成输入无法充分支持的性能、材质或适用性结论。正式结果保存在 `reports/generation/evaluation/baseline_metrics.json`。

上述 Week 1 正式指标使用的是确定性解码（`do_sample=false`）和 `baseline_v1` Prompt。原配置保存在 `configs/generation_greedy_baseline.json`，原始输出和人工标注不覆盖。现在的 `configs/generation.json` 改用采样解码（`do_sample=true`），因此 `temperature=0.7` 和 `top_p=0.9` 会真正传给模型；同时 `baseline_v2` Prompt 加强了“恰好三个卖点”的要求。生成代码仍保留 `baseline_v1` Prompt，便于复现旧基线。新设置的输出不能沿用旧人工评分；后续 LoRA 对比也应使用相同的解码与 Prompt 版本。

首次采样试跑仍用 `baseline_v1` Prompt，结果见 `reports/generation/baseline/smoke_test_sampling.json`：5条中只有3条符合三个卖点的结构要求，平均耗时53.9秒/条。随后在同一模型实例内对3条样本交替测试两种解码：确定性解码分别约5.1、6.2、6.3秒，采样分别约7.0、6.3、45.4秒；慢样本并非因为输出更长，根因尚未确定。这说明偶发耗时波动需要持续观察，不能把首次试跑的慢速直接归因于采样参数。对照原始记录在 `reports/generation/baseline/decoding_benchmark.json`。

只加强卖点数量约束、保持其他生成参数不变后，`baseline_v2` Prompt 的5条复测全部通过结构校验，平均6.056秒/条，见 `reports/generation/baseline/smoke_test_sampling_prompt_v2.json`。固定的100条 test 样本也已重新生成到 `reports/generation/baseline/evaluation_outputs_sampling_v2.json`：99条通过结构校验，1条仍生成4个卖点；平均6.430秒/条，P95为7.707秒，最慢8.798秒，模型加载12.266秒。结构成功率高于旧基线的96%，这次平均耗时低于PRD的12秒目标。新版文案后来使用独立标注表完成100条人工复核，没有沿用旧版评分。

## 100条基础评测

以下是已完成的历史确定性 Baseline 的运行顺序，用于说明结果来源。现有输出、标注和指标已经保留；不要直接重跑并覆盖这些文件。

```powershell
python scripts/prepare_generation_evaluation.py
python scripts/generate.py --config configs/generation_greedy_baseline.json --input data/processed/week1_v3/generation_evaluation_samples.jsonl --sample-count 100 --output reports/generation/baseline/evaluation_outputs.json --offline
python scripts/evaluate_generation.py --prepare
```

人工评测表是 `reports/generation/evaluation/annotation_pool.csv`。不要修改商品编号、输入属性和模型输出，只填写以下列：

- `matched_attribute_count`：输入属性中，有多少项在标题、卖点或短详情里被正确表达。意思相同的改写也算命中，但表达错误不能算。
- `fluency_pass`：三类文案整体语法通顺、逻辑自然填1，否则填0。
- `factual_error_count`：统计输入无法支持或与输入矛盾的独立事实数量；例如输入只有“塑料”，输出写“环保塑料、坚固耐用”属于事实外扩。
- `category_style_pass`：3C 是否侧重参数和功能、家居是否侧重生活场景和实用性，并且整体自然，符合填1，否则填0。
- `review_notes`：可选；有错误时建议简短写明问题，正确样本不必填写。

`auto_exact_matched_count` 只是程序找到的原样字符串数量，不能直接复制到人工命中数。`format_valid=0` 的4条样本是因为模型生成了4个而不是3个卖点，仍需正常评审其内容。

如果表中备注以 `AI初标` 开头，表示该行只是模型辅助初标。人工逐行确认或修改后，应删除该行备注开头的 `AI初标：`；问题说明本身可以保留。只要还有 AI 初标标记，默认评测命令就会拒绝生成正式人工指标，避免把辅助标签误当成人工结论。

全部100条人工复核、移除 AI 初标记并保存后运行：

```powershell
python scripts/evaluate_generation.py
```

脚本会计算核心属性命中率、通顺率、事实错误样本率、品类风格通过率，以及平均/P95生成耗时，并与 PRD 的75%、80%和12秒门槛进行对照。当前100条人工评分已完成，运行后可重建正式指标文件。

## 新版采样文案复核

新版沿用上面的同一批100条 test 商品和相同的人工评分规则，但使用独立文件，旧版标注与指标不变。当前复核表是 `reports/generation/evaluation/annotation_pool_sampling_v2.csv`，对应模型输出为 `reports/generation/baseline/evaluation_outputs_sampling_v2.json`，评测配置为 `configs/generation_evaluation_sampling_v2.json`。

这张表的四个人工评分列 `matched_attribute_count`、`fluency_pass`、`factual_error_count`、`category_style_pass` 曾按旧版口径填写 AI 初标，现已由用户逐条复核并清除全部 `AI初标：` 标记。`format_valid` 和 `auto_exact_matched_count` 是程序辅助列，不是人工评分。商品 `608516760684` 有四个卖点，表中保留了全部内容供复核。

用 Excel 打开 CSV 时，请通过“数据 → 自文本/CSV”导入，并把 `product_id` 列设为文本，避免保存时被改写成科学计数法；只编辑人工评分列和 `review_notes`。

如果需要重新建立一份尚不存在的新版复核表，运行：

```powershell
python scripts/evaluate_generation.py --config configs/generation_evaluation_sampling_v2.json --prepare
```

脚本会拒绝覆盖已经存在的标注表。100条全部复核并保存后，运行：

```powershell
python scripts/evaluate_generation.py --config configs/generation_evaluation_sampling_v2.json
```

人工复核完成后，已运行上述命令生成 `reports/generation/evaluation/baseline_metrics_sampling_v2.json`，旧版指标没有覆盖。Excel 保存时曾将94个商品编号改成科学计数法；修复前逐列核对了100条与原始模型输出的顺序、属性和文案，仅恢复 `product_id`，人工评分与备注保持不变。指标文件记录了最终标注表的 SHA-256 校验值。

| 同一批100条 test 商品 | 旧版确定性解码 + v1 Prompt | 新版采样解码 + v2 Prompt |
| --- | ---: | ---: |
| 核心属性命中率 | 88.00% | 88.16% |
| 通顺率 | 99% | 98% |
| 含事实错误的样本比例 | 56% | 73% |
| 结构成功率 | 96% | 99% |
| 平均生成耗时 | 6.443秒 | 6.430秒 |

新版属性命中率、通顺率和平均耗时均达到PRD对应门槛，但事实错误样本比例变高。这次同时改变了解码方式和 Prompt，属于**整体配置对比**，不能据此判断哪一个改动导致事实错误增多；采样也有随机性，表中结果仅代表各版本的一次生成运行。

新版已登记到 `reports/experiment_log.csv`，方法名为 `sampling_prompt_v2`，与旧版 `generation_baseline_v1` 使用相同评测集标识。复现登记命令：

```powershell
python scripts/track_experiment.py --module generation --method sampling_prompt_v2 --metrics reports/generation/evaluation/baseline_metrics_sampling_v2.json --manifest reports/generation/evaluation/annotation_manifest.json --config configs/generation.json --config configs/generation_evaluation_sampling_v2.json --notes "同一批100条测试商品；同时更改采样解码和提示词，不能归因于单一变量"
```

## 环境和运行

生成使用独立环境，不影响 PaddlePaddle 检索环境：

```powershell
conda create -n ecommerce-generation python=3.10 pip=25.2 -y
conda activate ecommerce-generation
conda env config vars set HF_HOME="E:\Projects\Baidu\.model-cache\huggingface" HF_HUB_CACHE="E:\Projects\Baidu\.model-cache\huggingface\hub" PIP_CACHE_DIR="E:\Projects\Baidu\.package-cache\pip"
conda deactivate
conda activate ecommerce-generation
python -m pip install -r requirements-generation.txt
python scripts/generate.py --sample-count 5 --output reports/generation/baseline/smoke_test_sampling_prompt_v2.json
```

第一次运行会下载模型到配置指定的 E 盘缓存。以后可以增加 `--offline`，确认只使用已经下载的文件：

```powershell
python scripts/generate.py --sample-count 5 --offline --output reports/generation/baseline/smoke_test_sampling_prompt_v2.json
```

旧的确定性试跑结果保存在 `reports/generation/baseline/smoke_test.json`；当前采样与新版 Prompt 的试跑保存在 `reports/generation/baseline/smoke_test_sampling_prompt_v2.json`。上面命令展示输出位置，但重新运行仍会覆盖同名的新报告；如需保留多次试验，应指定新的 `--output` 路径。报告保存真实模型输入、原始输出、解析结果、运行时间和 PyTorch 显存。`reference_title_not_given_to_model` 只用于结果对照，没有送入提示词。需要只测一条时省略 `--sample-count 5`。
