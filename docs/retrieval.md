# ERNIE-ViL跨模态检索与规则rerank技术说明

正式技术入口；历史及正式指标原样保留于reports/retrieval与formal_baseline_report。这里记录实际实现，不以PRD建议冒充代码功能。

## 1. 模型、环境与可复现限制

模型为PaddlePaddle/ernie_vil-2.0-base-zh（ERNIE-ViL 2.0中文Base）。环境ecommerce-retrieval：Python3.10.21，Paddle GPU3.3.0、PaddleNLP3.0.0b4、NumPy1.26.4、aistudio-sdk0.2.6、Faiss CPU1.15.0；正式Baseline归档记录Pillow12.3.0，requirements-data声明11.1.0，两者不混写。来源：requirements-retrieval.txt及[冻结Baseline环境](../reports/formal_baseline_report.md)。

缓存由PPNLP_HOME指向../.model-cache/paddlenlp。Windows初始化先调用src/retrieval/windows_cuda.py，补wheel内CUDA DLL搜索路径，不改变全局PATH。旧Paddle2.6.1缺系统cuDNN，新版PaddleNLP与aistudio版本兼容问题已有历史记录，不能无理由升级。

**当时没有记录不可变ERNIE-ViL revision或原权重完整SHA**。当前可看到缓存，但不能反向证明历史运行权重revision。正式特征/索引有hash；新接手应优先使用这些附件，不能声称重新下载main必然bitwise复现。

## 2. 特征提取：实际位置，不猜“第几层”

仓库封装src/retrieval/ernie_vil.py:ErnieVilEmbedder通过PaddleNLP Taskflow('feature_extraction', model=..., is_static_model=False)执行，读取返回features后做normalize_rows。没有指定隐藏层编号，不是自定义截取中间层。

本轮只读检查本机PaddleNLP3.0.0b4实现：
- taskflow/multimodal_feature_extraction.py动态图调用_model.get_image_features(pixel_values)或get_text_features(input_ids)；
- transformers/ernie_vil/modeling.py实际返回vision_outputs[1]与text_outputs[1]，即各encoder pooled output；未在该接口另写一个投影/中间层抽取。
- 文档示例把视觉pool称pooled CLS；但仓库未冻结具体N层编号和内部pooler配置，**当前工程材料中未找到可靠的历史层号记录，需要人工确认**。不能仅根据Base模型常见层数写“第12层”。
- 输出768维由既有feature_library/index_build实测记录支持；编码后L2归一化，零向量直接报错。

本机库源码及缓存预处理配置的路径/hash/提取依据附在[交付观察证据](../reports/delivery/runtime_observations.json)。这些是当前观察，不冒充Baseline运行时已经冻结的库源码证据。

## 3. 文本和图像预处理

商品文本由src/retrieval/feature_library.py:build_product_text按configs/retrieval.json的title、description顺序，strip后以换行连接、忽略空值。所有description为空，因此实际仅标题；没有用生成详情回填。文本源清洗NFKC/空白规范化在数据层；查询encode_texts只验证非空，不另写营销去除或分词策略。

PaddleNLP当前Taskflow默认max_length=128，padding=max_length、truncation=true；项目构造Taskflow未覆盖max_length。应与Qwen的2048 preflight不截断区分。是否历史框架恰好相同参数没有独立源码hash记录，当前按已记录依赖版本和本机源码观察解释。

图像：项目用Pillow完整读取并convert RGB；后续resize/crop/normalize交给AutoProcessor。当前本机ERNIE-ViL缓存preprocessor_config.json：
- 按shortest_edge=224保持宽高比resize，resample=3（Pillow bicubic）；
- center crop 224×224；
- convert RGB；像素rescale=1/255；
- mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]逐通道normalize。
- 这些实际值来自缓存配置，不从CLIP默认值猜测；历史预处理配置hash缺失，需区分“当前观察”与“当时冻结”。

## 4. 特征库、相似度和双向检索

| 资产 | 实际位置 |
|---|---|
| 商品库 | data/processed/week1_v3/multimodal/products.jsonl，1,992件 |
| 图片 / 文本向量 | artifacts/features/ernie_vil_week1_v3.npy / ernie_vil_week1_v3_text.npy |
| 融合向量（实验） | artifacts/features/ernie_vil_week1_v3_fused.npy |
| 图片 / 文本索引 | artifacts/indexes/ernie_vil_week1_v3.faiss / ernie_vil_week1_v3_text.faiss |
| 融合索引（实验） | artifacts/indexes/ernie_vil_week1_v3_fused.faiss |
| 行号映射 | artifacts/indexes/ernie_vil_week1_v3_metadata.jsonl |

build_index.py / build_text_features.py，batch_size=8、gpu:0，使用float32的归一化向量和Faiss IndexFlatIP。归一化后内积=cosine；1,992规模不需要近似索引。融合为normalize(0.5*text+0.5*image)，只是历史优化候选，不是正式纯跨模态Baseline。

T→I：查询中文text encoder → 图片索引。
I→T：查询图片vision encoder → 商品文本索引。
完整CLI商品库含所有project split，仅用于本地检索演示；正式test50只过滤test200候选，排除查询商品自身，不把1,992全库指标当test200指标。相关性/指标规则见[evaluation](evaluation.md)。

交付后演示（本轮不执行）：

~~~powershell
conda activate ecommerce-retrieval
$env:PPNLP_HOME = (Resolve-Path '../.model-cache/paddlenlp').Path
python -B -X utf8 scripts/search.py --text "USB接口的迷你有线键盘" --top-k 5 --json
python -B -X utf8 scripts/search.py --image data/images/week1_v3/16454614360.webp --top-k 5 --json
~~~

需要既有缓存、向量/索引/元数据附件。首次加载Taskflow若缓存不完整可能访问网络，当前没有强制离线快照revision的完整检索入口；接手者不能将其与生成离线保护混淆。

建库命令为python scripts/build_index.py与python scripts/build_text_features.py。默认拒绝覆盖，但存在--overwrite选项；本轮禁止重建冻结索引，不能把此选项放在快速启动流程中。历史smoke_test_retrieval.py为SMOKE，不是正式效果结论。

## 5. 冻结rerank：公式与信号

src/retrieval/rerank.py + configs/retrieval_rerank_formal_v1.json；规则词表从retrieval_rerank_validation_v1.json固定继承。

~~~text
final_score = similarity_score * category_factor
            + 0.24 * matched_constraint_ratio
            - 0.12 * conflicted_constraint_ratio
~~~

Top20原始候选重排后截Top10。二级同类factor=1.0，同一级=0.7，跨一级=0.4；未从query解析品类时factor=1.0。约束来自真实查询文字，候选支持来自标题/属性规范化文本，不用隐藏人工相关标签打分。matched/conflicted ratio分母是解析出的查询约束数，未知不算命中；负向概念冲突优先，数值按实际规则文本匹配。

相似度先按冻结6位精度；同分依次rerank score降序、原candidate rank升序、product ID升序。若formal Top20存在负相似度，乘小于1因子可能反向奖励，因此安全保护为中止，不clamp或临时调权。

只做Text→Image：图片查询无独立可用结构化信号，Image→Text严格passthrough Baseline Top10，不偷用查询商品隐藏品类。已知USB可能是无线接收器，冻结规则USB→wired过宽；报告保留此限制，不据test修词表。

**search.py没有接入此rerank；没有面向任意用户query的rerank CLI。** 正式入口run_retrieval_rerank_formal_test.py已经单次执行，仅作为可复现定位，不允许重跑。tune_retrieval_rerank.py是EXPERIMENT，不是在线推理入口。

## 6. 效果和限制

正式test50：
- T→I：P@10 0.436→0.518，R@10 0.612211→0.719621，MRR0.725270→0.804524，NDCG0.569573→0.730356。
- I→T：P@10 0.454、R@10 0.626766、MRR0.694524、NDCG0.582210，全部保持Baseline。
- T→I Recall达到PRD65%，Precision未达55%；图片方向未达两项。耳机MRR下降，不写所有类别指标都改善。
- validation24的Recall只是pooled；test50的同类候选全人工标注分母，定义见评测文档。
- 编码/搜索历史smoke约0.276秒/五条编码、0.002秒搜索，仅是链路观察；未找到独立50次端到端响应/冷启动正式性能验收，不宣称完整≤6秒达标。
- 原title可能含营销或歧义，description全空；相关性判断也受样本规模及人工主观性限制。

[正式报告](../reports/retrieval/rerank/formal_rerank_v1_test50_report.md)、[冻结manifest](../reports/retrieval/rerank/rerank_v1_formal_manifest.json)、[正式排名](../reports/retrieval/rerank/formal_rerank_v1_test50_rankings.csv)是追溯入口，不修改或覆盖。
