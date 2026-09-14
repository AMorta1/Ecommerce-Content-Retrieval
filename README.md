# Ecommerce-Content-Retrieval

电商内容智能生成与多模态检索算法验证项目。当前完成 Week 1 数据处理、ERNIE-ViL 2.0 中文版检索基线，以及 Qwen2.5-7B-Instruct 文案生成 Baseline 推理和100条人工评测。主要运行设备为本地 RTX 4070 Laptop（8GB 显存）。

## 当前可用数据

`week1_v3` 从两个大类、八个细分类各抽取 250 条商品；图片失败和重复过滤后为 **1,992 条**：训练 1,592、验证 200、测试 200。v2 人工抽查发现的笔记本内置替换键盘已经通过规则排除。

品类为耳机、键盘、鼠标、移动电源、保温杯、收纳箱、拖把、垃圾桶。类别规则仍需人工审核，原始属性也可能互相矛盾。它是基线实验子集，不是已经标注完成的 LoRA 或检索评测数据。

## 目录

```text
configs/                 数据、生成、检索和评测参数配置
src/data/                流式读取、属性解析、预处理、图片处理
src/generation/          Qwen 提示词、模型加载和生成评测逻辑
src/retrieval/           ERNIE-ViL 编码、商品信息读取、Faiss 检索
scripts/                 可以执行的入口
tests/                   数据、生成、检索和实验记录测试
data/processed/          当前实验 JSONL 和配置快照（不提交 Git）
data/images/             本地图片缓存（不提交 Git）
artifacts/               可重建的大文件：图片向量和 Faiss 索引（不提交 Git）
reports/                 小型实验结果、人工审核表和历史证据（提交 Git）
docs/                    数据、检索、生成和评测说明
```

## 运行数据处理

以下命令都在仓库根目录执行。原始三个 JSON 默认位于仓库上一级；修改 `configs/data.json` 可调整路径。原始文件、图片和处理后的全量 JSONL 不随仓库分发。

流式统计、清洗、划分只使用 Python 标准库；图片处理使用 Pillow。当前数据代码实际在 Python 3.13.9、Pillow 11.1.0 上验证。后续模型依赖和模型运行环境另行确定。

```powershell
python -m pip install -r requirements-data.txt
python scripts/inspect_data.py --input ../product1m_product5m_id_label.json --output reports/data/census
python scripts/prepare_data.py
python scripts/download_images.py --workers 4
python scripts/prepare_inference_samples.py
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

`prepare_data.py` 拒绝覆盖已有输出目录。重复实验使用新的目录，例如：

```powershell
python scripts/prepare_data.py --output data/processed/week1_repeat
python scripts/download_images.py --dataset data/processed/week1_repeat
```

图片下载会验证并复用当前版本的缓存，失败样本记录在 `image_manifest.jsonl`，不会填入假图片。可以通过 `--limit 5` 先选择少量训练商品测试下载流程。

当前数据和检索都默认读取 `week1_v3`。旧版大文件和重复报告已经清理，只在 `reports/history/` 保留发现品类问题所需的 v2 人工抽查证据。处理过程中的候选数据库会自动删除。

## 运行 ERNIE-ViL 检索试验

检索使用独立的 Python 3.10 环境，不影响数据处理环境。首次安装和首次运行会下载较大的 GPU 运行库与约 777MB 模型权重；它们保存在本机环境和 PaddleNLP 缓存中，不提交 Git。

```powershell
conda create -n ecommerce-retrieval python=3.10 pip=25.2 -y
conda activate ecommerce-retrieval
python -m pip install --use-pep517 -r requirements-retrieval.txt
python scripts/build_index.py
python scripts/smoke_test_retrieval.py
```

`build_index.py` 为 1,992 张商品图片提取向量，并建立精确余弦检索索引。默认不会覆盖已有输出；确实需要重建时使用 `--overwrite`。模型代码会自动处理 Windows 下 Paddle wheel 自带 CUDA DLL 的加载路径，不需要修改系统全局 PATH。

建立索引后，可以直接检索：

```powershell
python scripts/search.py --text "USB接口的迷你有线键盘" --top-k 5
python scripts/search.py --image data/images/week1_v3/16454614360.webp --top-k 5
```

当前实测为 768 维向量，完整图片库编码约 92 秒；五条文本批量编码约 0.276 秒，搜索全部候选约 0.002 秒。五条检查查询的 50 个 Top-10 结果都命中查询对应品类，但细粒度属性并不稳定。这些查询不是独立人工相关性标签，因此只能作为基础效果观察，不能当作正式 Recall@10 或 Precision@10。

模型权重、图片向量和索引是本地大文件，不提交 Git。Leader 拿到代码和原始数据后运行建索引命令即可复现；这是推理预计算，不是重新训练模型。如需免去这一步，可以把 `artifacts/features/` 和 `artifacts/indexes/` 作为单独实验附件发送。

## 准备检索人工评测

第一版 validation 评测包含 24 条查询和每条 20 个候选，共 480 行，现已全部完成人工复核。标注表位于：

```text
reports/retrieval/evaluation/annotation_pool.csv
```

运行正式评测：

```powershell
python scripts/evaluate_retrieval.py
```

当前基线 Precision@10 为 49.58%，低于 PRD 参考目标 55%；候选池 Recall@10 为 67.08%。Recall 只能基于已检查候选池计算，因此明确写为 `pooled_recall_at_10`，不能冒充覆盖全库的完整 Recall。

如果审核 CSV 被 Excel 保存成科学计数法，可用 `scripts/repair_csv.py` 检查，并增加 `--repair` 从对应图片文件名恢复原始商品编号；人工填写的审核结果不会被清除。

## 运行 Qwen 文案生成试验

文案生成使用独立的 `ecommerce-generation` 环境。模型权重固定缓存在仓库上一级的 `.model-cache/huggingface/`，不进入 Git；推理时使用 bitsandbytes NF4 4-bit，以适配8GB显存。第一次运行会下载模型，后续增加 `--offline` 可强制只读本地缓存。

```powershell
conda create -n ecommerce-generation python=3.10 pip=25.2 -y
conda activate ecommerce-generation
python -m pip install -r requirements-generation.txt
python scripts/generate.py --sample-count 5
python scripts/generate.py --sample-count 5 --offline
```

真正送入模型的只有 `generation_input` 中的品类和筛选后属性，不包含作为对照的原始标题。输出固定为生成标题、三个卖点和短描述。完整的环境变量设置、输入边界和报告解释见 `docs/generation.md`。

当前五条可行性样本均成功生成并通过 JSON 结构校验。内容仍存在把普通材质扩写成“环保”、把普通键盘写成“游戏键盘”等事实外扩，后续需通过提示词或微调优化。

PRD 要求从 test 集抽取100条人工评估核心属性命中率和通顺度。正式评测池按两大类各50条分层抽样并完成人工复核：核心属性命中率88%，通顺率99%，均通过75%和80%的门槛；平均生成6.443秒，也通过12秒门槛。严格结构成功率为96%，事实错误样本率为56%，说明 Baseline 虽达到PRD基础门槛，但事实约束仍是后续优化重点。标注表和正式指标分别位于 `reports/generation/evaluation/annotation_pool.csv`、`reports/generation/evaluation/baseline_metrics.json`，填写方法和复现命令见 `docs/generation.md`。

## 记录和对比实验版本

`scripts/track_experiment.py` 将生成、检索的正式指标登记到同一张 `reports/experiment_log.csv`。每行代表一个实验版本，包含模型、方法、固定评测集标识、完整配置快照、关键指标和相对 Baseline 的变化。当前已登记两个 Week 1 Baseline。

登记生成实验：

```powershell
python scripts/track_experiment.py --module generation --method baseline --metrics reports/generation/evaluation/baseline_metrics.json --manifest reports/generation/evaluation/annotation_manifest.json --config configs/generation.json --config configs/generation_evaluation.json
```

登记检索实验：

```powershell
python scripts/track_experiment.py --module retrieval --method baseline --metrics reports/retrieval/evaluation/baseline_metrics.json --manifest reports/retrieval/evaluation/annotation_manifest.json --config configs/retrieval.json --config configs/retrieval_evaluation.json
```

后续版本将 `--method` 改为 `lora`、`rag` 或 `rerank`，并传入该版本独立的指标、清单和配置文件。同一模块和版本会更新原行，不会重复追加；只有评测集标识相同的版本才会自动计算差值。优化版指标不能覆盖现有 Baseline 文件。

## 后续模型验证读取位置

```text
data/processed/week1_v3/multimodal/train.jsonl
data/processed/week1_v3/multimodal/validation.jsonl
data/processed/week1_v3/multimodal/test.jsonl
data/processed/week1_v3/inference_samples.jsonl
```

最后一个文件是经过图片和文本检查的五条样本，包含：

- `generation_input`：明确筛选后的品类和属性，排除已发现冲突的字段；不包含目标标题。
- `image_path`：相对仓库根目录的有效图片路径。
- `smoke_query`：用于试跑文本检索的描述，不是人工相关性标签。
- `sample_review`：已知数据问题及检查方式。

JSONL 中仍保留原始标题、原始属性和来源。看图检查不等于验证了实际商品参数。推理验证样本只保留上述一份，由 `prepare_inference_samples.py` 生成。

## 文档与结果

- [数据处理与结果](docs/data.md)
- [模型选型与 ERNIE-ViL 检索基线](docs/retrieval.md)
- [Qwen 文案生成基线与人工评测](docs/generation.md)
- [检索人工评测说明](docs/evaluation.md)
- [多版本实验汇总](reports/experiment_log.csv)
- [v3 全量索引构建记录](reports/retrieval/baseline/index_build.json)
- [v3 ERNIE-ViL 五样本试跑结果](reports/retrieval/baseline/smoke_test.json)
- [全量标签频数](reports/data/census/label_counts.csv)
- [v3 独立校验结果](reports/data/validation.json)
- [v3 的 40 条抽查结果](reports/data/category_review.csv)
- [v3 近重复与同型号检查](reports/data/duplicate_audit.json)
- [v2 原始抽查证据](reports/history/category_review_week1_v2.csv)
