# 内容生成、RAG与LoRA技术说明

最终技术入口，合并原生成说明中分散的运行/版本信息；**不改实验协议、Prompt源码、config或正式报告**。质量结果见[项目简报](project_brief.md)，运行导航见[README](../README.md)，指标定义见[evaluation](evaluation.md)。

## 1. 模型、环境与加载边界

| 项目 | 已有工程证据 |
|---|---|
| 基座 | Qwen/Qwen2.5-7B-Instruct |
| RAG formal / LoRA及后续固定revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| 本机snapshot | ../.model-cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 |
| Baseline历史revision | 当时未记录；不能以当前缓存revision反向证明 |
| 正式LoRA训练环境 | Python3.10.21 / Torch2.12.0+cu130 / Transformers5.16.1 / bitsandbytes0.50.2 / Accelerate1.14.0 / PEFT0.21.1 / safetensors0.8.0 / CUDA13.0 |
| 训练随机性限制 | seed固定，但deterministic_algorithms=false；不承诺跨GPU/库bitwise相同 |
| 量化 | bitsandbytes NF4 4-bit、double quant=true、compute BF16 |
| 存储精度 | 省显存路径冻结embedding/head为BF16，norm及可训练LoRA为FP32 |
| 加载 | 固定本地snapshot、local_files_only=True；正式入口设置HF_HUB_OFFLINE和TRANSFORMERS_OFFLINE，不做网络fallback |

来源：[训练final manifest](../reports/generation/lora/lora_v1_training/final_manifest.json)、[pretraining manifest](../reports/generation/lora/lora_v1_pretraining_manifest.json)、[P7协议](lora_project_validation_v1_protocol.md)。snapshot绝对路径只在报告作本机观察，运行从config的相对cache路径解析。

requirements-generation.txt记录四个推理依赖，不是完整LoRA锁文件：PEFT安装为经用户确认的--no-deps peft==0.21.1；没有安装trl/datasets。历史freeze与requirements不一致不能靠文档声称已经自动解决。工作簿的openpyxl在基础Python中，不是模型推理环境必备。本轮不安装或改依赖。

## 2. 输入、Prompt和推理配置

### 单调用Baseline

src/generation/qwen.py的build_messages / QwenGenerator；configs/generation.json：
baseline_v2，max_input_tokens=2048，max_new_tokens=320，do_sample=true，temperature=0.7，top_p=0.9，seed=42。输入generation_input只有品类+冻结筛选属性，原标题仅在报告中作参考，不进入Prompt。一次调用输出JSON：title、恰好三条selling_points、short_description。

模型tokenizer.apply_chat_template生成真实输入；当前代码preflight不截断，超限报错。历史Baseline没有完整不可变revision记录，复现旧版本应定位beaf9d1代码及formal_baseline_manifest，而不是假定当前源码等于历史版本。旧greedy配置只是LEGACY，不是最终Baseline。

generate.py的计时来自调用前后perf_counter，加载另计；报告generation_average还含该循环的解析/记录开销，不能当服务端端到端响应时间。旧源码未使用P7式显式CUDA同步；不同调用结构不作速度因果比较。

交付后非正式演示命令（本轮不执行，输出必须是新文件）：

~~~powershell
python -B -X utf8 scripts/generate.py --config configs/generation.json --input data/processed/week1_v3/inference_samples.jsonl --sample-count 1 --offline --output reports/delivery_user_demo_baseline.json
~~~

load_samples要求generation_input与product_id，不能直接把原train.jsonl当这条CLI输入。generate.py会覆盖同名文件，README已提示需先检查。正式100件已完成，禁止再次把generation_evaluation_samples传入重跑。

### 三任务Base/LoRA（冻结公平对照）

src/generation/lora_data.py:build_instruction_text为任务模板；src/generation/lora_validation.py负责真实chat模板、token preflight、调用、parse_task与assemble。

| 参数 | 冻结值 / 来源 |
|---|---|
| tasks | title / selling_points / short_description，三次独立调用 |
| 输入 | 品类+P5可靠核心属性，再叠加原源审计门控；源title不输入 |
| 模板 | lora_copy_templates_v1；system及三个output要求在P7 config中 |
| 输入tokens | ≤2048，truncation=false；完整核心JSON必须实际进入token上下文 |
| 新生成tokens | 每task最多320 |
| decoding | do_sample=true、temperature0.7、top_p0.9、top_k20、repetition_penalty1.05、num_beams1 |
| 推理工程 | use_cache=true，gradient_checkpointing=false，NF4/double quant/BF16 |
| seed | int(SHA256('p7_lora_v1:42:product_id:task_type')前8hex,16) mod2147483647，不含模型版本 |
| 实际默认生成参数 | inference_base.json / inference_lora.json中的decoding_resolved，不仅抄config摘要 |
| latency | 每task CUDA同步后perf_counter包围generate，三任务总数为三次latency之和；不含模型加载/检索/完整UI |

Base和LoRA输入、Prompt、长度、解码、seed与工程准备一致，唯一模型差别是adapter加载。不能拿单调用旧Baseline Prompt与三任务LoRA比较后把变化全部归因LoRA。独立任务原文及组装全文均保存，解析失败原文仍人工评审。

[P7协议](lora_project_validation_v1_protocol.md)、[test100协议](lora_project_test100_v1_protocol.md)、configs/lora_project_validation_v1.json以及project_test100_v1/inference_*.json为依据。实际代码保持所有200件project_validation生成，但人工仅冻结quick32；不是200件全人工复核。

## 3. LoRA数据、训练和低显存实现

P5从week2_train_v1构造，不使用Baseline/RAG/模型输出当target。title是可靠组件按原生标题顺序重组，不是原生标题全文自由润色；卖点固定3条中性字段值，短详情最多5属性。原标题只用于target顺序、风格和审计。事实约束是输入机制职责，LoRA目标为表达结构；模板target偏机械也是自然度限制来源。完整数据规则见[data](data.md)。

正式配置configs/lora_qlora_train_v1.json已冻结；其中status仍是当时pending候选，不代表今天没训练，真实完成状态由final_manifest判定。

| 训练参数 | 实际冻结值 |
|---|---|
| LoRA层 | q_proj / v_proj，CAUSAL_LM，bias=none |
| rank / alpha / dropout | 8 / 32 / 0.05；可训练2,523,136参数 |
| micro batch / accumulation | 1 / 8；不足8条的尾组按实际条数归一化 |
| 最大序列长度 | 320；不截断、不固定长padding；P5实际最长269 |
| optimizer | torch.optim.AdamW，LR2e-4，betas0.9/0.999，eps1e-8，weight_decay0.01 |
| scheduler / warmup / grad clip | constant_lambda_1 / 0 / max_grad_norm=null |
| epochs / seed | 3 / 42；逐epoch Python随机打乱 |
| gradient checkpointing / use_reentrant | true / false；训练use_cache=false |
| early stopping | 不启用 |
| dev | 全402条lora_train_dev，no_grad，参数更新0，保持工程配置及RNG |
| evaluation / best | 每epoch结束；仅完整dev loss最低的epoch-end checkpoint能更新best |
| checkpoint | 每151 optimizer steps及epoch结束，保留latest+best；不保存完整7B基座 |
| 状态保存 | adapter、optimizer、scheduler、epoch/global step/cursor、Python/Torch CPU/CUDA RNG、config与校验manifest |

每epoch 3,618 micro steps，ceil(3618/8)=453 optimizer steps，三epoch总1,359。latest为最近完整checkpoint；best只能来自已完成dev evaluation的epoch-end，不能拿未评估中间checkpoint参与比较。fresh formal必须全新adapter，不继承smoke/acceptance；resume只用于真实中断。

src/generation/qlora_memory.py的省显存实现：
- PEFT准备时暂把冻结的大embedding/head移到CPU，避免临时FP32副本撑爆显存，然后恢复BF16存储。
- 原形状BF16 head GEMM仍保留，只对assistant有效token logits按16-token片段转FP32计算CE，并使用checkpoint。
- 保留causal shift、-100屏蔽和有效token平均分母；不是减少rank/样本或改变监督目标。
- epoch train/dev loss为每条assistant-token mean CE的样本平均，不是全库token总数加权平均，也不是生成质量指标。

正式训练**已经完成，不要重跑**。以下只是识别历史命令：

~~~powershell
python scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode train --confirm-formal-training
# resume仅真实中断才使用；当前已完成，不能据此重启：
# python scripts/train_lora.py --config configs/lora_qlora_train_v1.json --mode train --confirm-formal-training --resume artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359
~~~

训练99.12分钟，峰值allocated5661.9MiB、dev5535.6MiB；epoch dev loss为0.019772963、0.017524495、0.013250582。best=epoch3 / checkpoint-step-001359。这些只说明训练target拟合，不直接说明表达收益。

### 最终adapter与加载入口

最终目录：artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter；adapter_model.safetensors为10,107,280 bytes，SHA256=890a061812321ccf6bbf598b37617fc59dcde70d57042c4304c219a0f6740e31。adapter_config.json、best.json/latest.json及checkpoint完整状态都需单独备份，不能只交一份权重。

实际加载链：固定snapshot AutoModelForCausalLM+BitsAndBytesConfig → prepare_kbit_with_cpu_staged_large_layers → PeftModel.from_pretrained(..., is_trainable=False) → eval/no_grad。正式推理流程位于lora_validation.py:generate以及lora_formal_test.py:generate；独立重新加载已由verify_lora_v1_best.py验收，报告best_adapter_reload.json已冻结，不再覆盖。

**通用单商品LoRA推理CLI当前缺失。** 上述generate是冻结数据scope入口，verify脚本固定train小样本，不能承诺“给任意input就运行”。新可复用CLI仅建议另行确认，无本轮代码实现。

## 4. RAG：实际实现，不把设计建议当已实现组件

src/generation/rag.py / qwen.py / grounding.py和scripts/build_rag_facts.py。知识单元来自同商品ID的结构化属性+标准品类，不注入原标题、主图推导、其他商品、模型输出或人工标签。主图/标题只作为质量审计证据。

实际是轻量的同商品过滤、字段优先级与跨任务RRF排序；**没有实现通用BM25+向量Retriever、Chroma或LangChain链**，不能因PRD建议就写已接入这些库。RRF为各task的1/(60+rank)求和，global共享Top-K=3，并按冻结tie-break规则稳定排序。

| 区域 | 规则 |
|---|---|
| Identity | category_l1/l2、可信品牌/型号；不占Top-K；核心身份标记mandatory coverage但不重复注入 |
| Mandatory Reliable Core | v4中源存在、eligible、单值/规范等价、正向/中性可表达核心事实；与品类核心清单相交 |
| Supplemental Top-3 | 剩余可用补充事实经跨task RRF选择；不是把全部源字段塞入输入 |
| Negative Constraints | 可靠无/否/不支持/有线等只读约束，不要求主动卖点覆盖，不占补充Top-K |

RAG v1 formal：rag_v6+rag_fact_policy_v3，Identity+全局Top3+Negative，不含独立Mandatory四区。RAG v2：policy_v4+四区，恢复覆盖，但P2.6未通过通顺门槛，因此没有原RAG v2 formal或P3结果。

v4例外：保温杯“大众/年代人群”“日常送礼/通用”和垃圾桶“家庭使用”只supplemental，不强制Mandatory；负向核心在Negative； distinct多值SKU暂缓；收纳箱净重需正数且kg/g。完整白名单与任务优先级在rag_fact_policy_v4.json，非所有core字段都无条件Mandatory。

REVIEW/CONFLICT字段处理见[data](data.md)。RAG与P5允许集合的细则不同，不能替换输入门控。模型prompt/preflight对所有序列化Mandatory事实在tokenizer后做实际tokens完整性检查；任何溢出/缺失直接报错，不裁剪，不报虚假的100%。missing源字段不计注入失败，正式命中分母不变。

Grounding是确定性词/值/单位/范围/身份规则线索，不是人工事实真值分类器。FAIL不自动等于事实错误，supported paraphrase可为false positive；PASS也不保证无隐含扩写。事实错误按完整可靠源证据复核，而不是只比较Top3或输入子集。

## 5. LoRA+RAG组合与运行保护

组合复用固定epoch3 adapter、三任务模板、解码、seed和量化配置；原可靠核心JSON保留，追加四区事实；system由“可靠核心属性”放宽为“可靠属性”是已冻结必要接口适配。它是上下文+接口联合处理，不是纯检索算法因果隔离。

configs/lora_rag_v2_development24_v1.json与lora_rag_project_test100_v1.json、src/generation/lora_rag_integration.py和lora_rag_project_test.py为实现。组合不得恢复原质量门控撤回核心；同商品事实不能跨商品借用。test身份阻断时四区为空，仍保留原core分母。

正式入口scripts/run_lora_rag_project_test.py --mode prepare/generate/review/summarize为一次性实验；300调用已完成，后续人工解盲已归档，**不能重新执行generate或summarize覆盖现有结果**。scripts/run_lora_rag_integration.py仅development24实验，不是产品推理入口。

结果归档：
- [LoRA训练报告](../reports/generation/lora/lora_v1_training/final_training_report.md)与final_manifest记录代码/config/data/env/base/hash。
- [Base/LoRA正式test](../reports/generation/lora/project_test100_v1/final_test_analysis.md)及human_review_finalization_manifest。
- [组合最终报告](../reports/generation/lora_rag/lora_rag_project_test100_v1/final_review_report.md)及final_review_manifest。
- [原RAG v2结束报告](../reports/generation/rag/rag_v2_p2_phase_summary.md)，32holdout未执行；不要误用组合实验将其改为P3完成。

代码+Git HEAD不能代表全部未提交工作区，正式协议同时归档原始代码字节与逐文件hash。历史pending config/automatic report保留当时状态；最终manifest决定最终状态。当前说明不追写历史精确revision，也不更改任何冻结参数。
