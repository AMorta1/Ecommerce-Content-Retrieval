# 模型选型与 ERNIE-ViL 检索基线

## 一句话结论

检索主线已经使用 `PaddlePaddle/ernie_vil-2.0-base-zh` 在本机 RTX 4070 Laptop 8GB 显存上跑通，并完成 `week1_v3` 全部 1,992 件商品的图片特征与标题文本特征库。当前 Baseline 采用纯跨模态策略：中文文本检索商品图片索引，本地图片检索商品文本索引。实验性图文融合特征仅保留作后续优化对照，不作为 Baseline。

24条 validation 查询、每条20个候选已经完成人工相关性复核。基线 Precision@10 为49.58%，低于PRD参考目标55%；候选池 Recall@10 为67.08%。由于没有标注 validation 全库，只能将后者称为 pooled Recall，不能冒充覆盖全库的完整 Recall。

## 为什么选择 ERNIE-ViL

| PRD 候选 | 当前结论 |
| --- | --- |
| ERNIE-ViL 2.0 Base | 官方中文模型已经在本机 GPU 跑通，作为检索基线 |
| CLIP ViT-B/32 | 原版主要面向英语，保留为备选或对照，不默认替代中文模型 |
| ERNIE-3.0-8K-Base | 未确认到可直接使用的准确公开生成权重，没有拿名字相近的理解模型替代 |
| Qwen2.5-7B-Instruct | 符合中文、参数量、框架和上下文要求，已作为文案生成基线；详见 `docs/generation.md` |

大白话：ERNIE-ViL 会把图片和中文描述都转换成 768 个数字。两组数字越接近，模型认为图片和文字越相关。它负责“找商品”，不负责“看图写文案”。

官方模型入口为 [ERNIE-ViL 2.0 中文版](https://paddlenlp.readthedocs.io/zh/latest/website/PaddlePaddle/ernie_vil-2.0-base-zh/index.html)。CLIP 的语言限制可见 [官方模型卡](https://github.com/openai/CLIP/blob/main/model-card.md)。

## 本机环境

- Windows，NVIDIA GeForce RTX 4070 Laptop GPU，约 8GB 显存。
- 独立 Conda 环境 `ecommerce-retrieval`，Python 3.10.21。
- PaddlePaddle GPU 3.3.0、PaddleNLP 3.0.0b4、NumPy 1.26.4、aistudio-sdk 0.2.6、faiss-cpu 1.15.0。
- 模型缓存由 `PPNLP_HOME` 指向 `E:\Projects\Baidu\.model-cache\paddlenlp`，不提交 Git。

旧版 Paddle 2.6.1 在当前 Windows 环境缺少系统级 cuDNN DLL，最终改用 Paddle 3.3.0。项目通过 `src/retrieval/windows_cuda.py` 在当前进程补全 wheel 自带 DLL 的搜索路径，不修改系统全局 PATH。PaddleNLP 3.0.0b4 与新版 aistudio-sdk 存在接口兼容问题，因此依赖文件固定为 aistudio-sdk 0.2.6。

## 基线实现

~~~text
1,992 张商品图 → ERNIE-ViL 图片编码 → 图片索引
                                      ↑
中文查询 → ERNIE-ViL 文本编码 ────────┘

1,992 条商品标题 → ERNIE-ViL 文本编码 → 文本索引
                                        ↑
查询图片 → ERNIE-ViL 图片编码 ──────────┘
~~~

- 商品库：`data/processed/week1_v3/multimodal/products.jsonl`。
- 图片向量：`artifacts/features/ernie_vil_week1_v3.npy`。
- 文本向量：`artifacts/features/ernie_vil_week1_v3_text.npy`。
- 实验性融合向量：`artifacts/features/ernie_vil_week1_v3_fused.npy`，只供后续优化对照。
- 图片、文本和实验性融合 Faiss 索引：`artifacts/indexes/` 下对应的三个 `.faiss` 文件。
- 行号与商品信息对应表：`artifacts/indexes/ernie_vil_week1_v3_metadata.jsonl`。
- 模型、批量大小和所有路径：`configs/retrieval.json`。

图片和文本向量先做 L2 归一化，再用 Faiss `IndexFlatIP` 排序；此时内积等价于余弦相似度。当前只有 1,992 个候选，CPU 精确索引已经足够快，没有必要让近似索引增加复杂度。

商品库包含 train、validation、test，是为了提供完整本地检索体验。已完成人工复核的 24 条 validation 文本查询继续作为开发指标。另从冻结 test 中分层选出了 50 件查询商品；正式 test50 查询集目前为 v3。在首次正式 test 指标产生前，用户确认替换两条无相关候选的查询；test200 划分、其余 48 条查询和 1,152 行人工标签不变。新查询的 48 行也已人工复核，正式 test50 纯跨模态指标已生成，详情见 `docs/evaluation.md`。

当前来源的 `description` 全为空，所以 PRD 建议的“标题+短描述”文本特征实际只使用标题。没有用模型生成的短描述回填，因为那会额外引入生成模型变量。该数据限制和特征哈希记录在 `reports/retrieval/baseline/feature_library.json`。

## 本机实测

| 项目 | 结果 |
| --- | ---: |
| 商品图片 | 1,992 张 |
| 批量大小 | 8 |
| 向量维度 | 768 |
| 模型加载 | 约 5.8–6.2 秒 |
| 完整图片编码 | 92.038 秒 |
| 平均每张图片编码 | 46.204 毫秒 |
| Faiss 建索引 | 0.029 秒 |
| 模型加载后分配显存 | 777.2 MiB |
| 五条中文查询编码 | 0.276 秒 |
| 五条查询搜索全部候选 | 0.002 秒 |

五条试跑查询覆盖耳机、键盘、鼠标、保温杯和垃圾桶。50 个 Top-10 候选都属于对应二级品类，说明模型能抓住粗粒度商品语义。

同编号目标商品图的排名为 124、86、128、19、159。这个结果不表示前十都不相关，而是说明“找到同类商品”比“精确找到指定 SKU 图片”容易。已观察到的典型 Bad Case 是：查询“USB 接口的迷你有线键盘”时，前列结果出现无线或蓝牙键盘，说明模型对连接方式等细粒度属性不稳定。

机器可读记录位于 `reports/retrieval/baseline/`。

## 运行方式

首次准备独立环境：

~~~powershell
conda create -n ecommerce-retrieval python=3.10 pip=25.2 -y
conda activate ecommerce-retrieval
python -m pip install --use-pep517 -r requirements-retrieval.txt
~~~

建立索引、跑五样本检查并实际检索：

~~~powershell
python scripts/build_index.py
python scripts/build_text_features.py
python scripts/smoke_test_retrieval.py
python scripts/search.py --text "USB接口的迷你有线键盘" --top-k 5
python scripts/search.py --image data/images/week1_v3/16454614360.webp --top-k 5
~~~

`build_index.py` 默认不覆盖已有特征和索引。数据或模型配置变化后，明确使用：

~~~powershell
python scripts/build_index.py --overwrite
~~~

模型权重、图片向量和索引不提交 Git。Leader 获取代码和原始数据后，需要运行一次建索引命令；这是推理预计算，不是重新训练模型。若希望对方直接运行，也可以把 `artifacts/features/` 和 `artifacts/indexes/` 作为单独附件交付。

## 当前交付与限制

已经完成：

- ERNIE-ViL GPU 加载和统一图文编码封装。
- 全部商品图片、标题文本特征库及 Faiss 索引；实验性融合索引保留作后续对照。
- 中文文本检索、图片检索和 Top-K 输出入口。
- 五样本链路检查、速度记录和可复现环境。
- validation 的24条查询、480行候选人工复核。
- Precision@10、候选池 Recall@10、MRR@10 和 NDCG@10 正式基线报告。
- 从冻结 test 分层选择 50 件查询商品，生成 1,200 行相关性标注表；v3 两处换选有记录和相关候选预检。
- test50 的 1,200 行人工相关性复核及文本搜图、图片搜文本正式基线评测；指标见 `reports/retrieval/test_evaluation/baseline_metrics.json`。

尚未完成：

- 类别重排、RAG 和优化前后对比。
- Gradio Demo、压力测试和最终优化方案的 test 对比评测。
- LoRA/QLoRA 训练闭环与微调效果验证。
