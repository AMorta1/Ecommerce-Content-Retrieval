"""Qwen 文案生成、输入约束与结构化输出解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT_FIELDS = ("generated_title", "selling_points", "short_description")
CATEGORY_STYLE_GUIDANCE = {
    "3C数码": "表达简洁专业，优先准确呈现已提供的型号、参数、连接方式和功能，体现科技产品的信息感。",
    "家居日用": "表达自然生活化，优先准确呈现已提供的材质、规格、适用场景和实用特点。",
}
RAG_STYLE_GUIDANCE = {
    "3C数码": "使用简洁、中性、参数化表达，只直接列出上下文已有的型号、参数、连接方式和功能字段值。",
    "家居日用": "使用简洁、中性表达，只直接列出上下文已有的材质、规格、结构和功能字段值。",
}
CONTENT_REQUIREMENTS = {
    "标题": "生成一个简洁商品标题，突出品类和最重要的已知属性，不堆砌关键词。",
    "核心卖点": "生成三个核心卖点，每个卖点是一句话，且每句话都必须能由输入属性直接支持。",
    "短详情": "生成一段简短、客观、连贯的商品描述，不添加输入中没有的性能、效果或适用对象。",
}


def validate_generation_input(generation_input: dict[str, Any]) -> None:
    """检查真正送入模型的字段，避免把目标标题泄露给模型。"""
    if "title" in generation_input:
        raise ValueError("generation_input 不能包含目标标题 title。")
    for field in ("category_l1", "category_l2", "attributes"):
        if field not in generation_input:
            raise ValueError(f"generation_input 缺少字段：{field}")
    if not isinstance(generation_input["attributes"], dict):
        raise TypeError("generation_input.attributes 必须是对象。")
    if generation_input["category_l1"] not in CATEGORY_STYLE_GUIDANCE:
        raise ValueError(f"不支持的一级品类：{generation_input['category_l1']}")


def build_messages(
    generation_input: dict[str, Any], prompt_version: str = "baseline_v1"
) -> list[dict[str, str]]:
    """按指定版本把商品属性转换为 Qwen 消息，保留旧基线 Prompt。"""
    validate_generation_input(generation_input)
    if prompt_version not in ("baseline_v1", "baseline_v2"):
        raise ValueError(f"不支持的 Prompt 版本：{prompt_version}")
    product_facts = {
        "一级品类": generation_input["category_l1"],
        "二级品类": generation_input["category_l2"],
        "商品属性": generation_input["attributes"],
    }
    style_guidance = CATEGORY_STYLE_GUIDANCE[generation_input["category_l1"]]
    content_requirements = CONTENT_REQUIREMENTS.copy()
    if prompt_version == "baseline_v2":
        content_requirements["核心卖点"] = (
            "恰好生成三个核心卖点，每个卖点是一句话，且必须能由输入属性直接支持；"
            "即使有更多属性，也只选最重要的三点，不得添加第四个卖点。"
        )
    task_requirements = "\n".join(
        f"- {content_type}：{requirement}"
        for content_type, requirement in content_requirements.items()
    )
    system_prompt = (
        "你是专业的中文电商文案运营师。只能依据用户提供的商品事实写作，不得编造参数、"
        "功能、认证、促销、销量或售后承诺。只输出合法 JSON，不要输出 Markdown 和解释。"
    )
    if prompt_version == "baseline_v2":
        system_prompt += "selling_points 数组必须恰好有三个字符串，不能多也不能少。"
    user_prompt = (
        "请一次生成商品标题、核心卖点和短详情。三个内容类型分别遵循以下模板要求：\n"
        f"{task_requirements}\n"
        f"品类风格：{style_guidance}\n"
        "输出 JSON 字段必须为 generated_title、selling_points、short_description，"
        "其中 selling_points 必须是包含三个字符串的数组。\n"
        f"商品事实：\n{json.dumps(product_facts, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_rag_messages(
    generation_input: dict[str, Any],
    rag_context: dict[str, Any],
    prompt_version: str = "rag_v1",
) -> list[dict[str, str]]:
    """构造只包含合格身份信息和全局共享 Top-K 事实的 RAG Prompt。"""
    validate_generation_input(generation_input)
    if prompt_version not in {"rag_v1", "rag_v2", "rag_v3", "rag_v4", "rag_v5", "rag_v6"}:
        raise ValueError(f"不支持的 RAG Prompt 版本：{prompt_version}")
    identity_facts = rag_context.get("identity_facts")
    selected_facts = rag_context.get("selected_facts")
    negative_constraint_facts = rag_context.get("negative_constraint_facts", [])
    if not isinstance(identity_facts, list) or not isinstance(selected_facts, list):
        raise TypeError("rag_context 必须包含 identity_facts 和 selected_facts 数组。")
    if not isinstance(negative_constraint_facts, list):
        raise TypeError("rag_context.negative_constraint_facts 必须是数组。")
    if prompt_version == "rag_v6" and "negative_constraint_facts" not in rag_context:
        raise ValueError("rag_v6 必须提供独立的 negative_constraint_facts。")
    if len(selected_facts) > int(rag_context.get("top_k", -1)):
        raise ValueError("RAG 注入事实数量超过配置的 Top-K。")
    all_facts = [*identity_facts, *selected_facts, *negative_constraint_facts]
    if any(fact.get("quality_status") != "eligible" for fact in all_facts):
        raise ValueError("RAG Prompt 只能注入 quality_status=eligible 的事实。")
    if any(fact.get("product_id") != rag_context.get("product_id") for fact in all_facts):
        raise ValueError("RAG Prompt 中存在跨商品事实。")
    if prompt_version == "rag_v6":
        selected_ids = {fact.get("fact_id") for fact in selected_facts}
        negative_ids = {fact.get("fact_id") for fact in negative_constraint_facts}
        if selected_ids & negative_ids:
            raise ValueError("rag_v6 内容 Top-K 与只读否定约束不能重复。")

    def is_negative_constraint(fact: dict[str, Any]) -> bool:
        field = str(fact["canonical_field"])
        values = [str(value).strip() for value in fact["normalized_values"]]
        negative_values = {"无", "否", "不支持", "没有", "未配备"}
        if any(value in negative_values or "不支持" in value for value in values):
            return True
        return field == "是否无线" and any(value == "有线" for value in values)

    def prompt_fact(
        fact: dict[str, Any], *, mark_negative: bool = False, mark_category: bool = False
    ) -> dict[str, str]:
        prompt_value = {
            "字段": str(fact["canonical_field"]),
            "值": "、".join(str(value) for value in fact["normalized_values"]),
        }
        if mark_category and fact["canonical_field"] in {"category_l1", "category_l2"}:
            prompt_value["约束"] = "仅分类标签；禁止推导适用对象、用途或使用场景"
        elif mark_negative and is_negative_constraint(fact):
            prompt_value["约束"] = "否定事实；可以不写成卖点，但禁止生成相反内容"
        return prompt_value

    marks_negative_constraints = prompt_version in {"rag_v4", "rag_v5"}
    marks_category_constraints = prompt_version == "rag_v5"
    prompt_context = {
        "必要身份信息": [
            prompt_fact(fact, mark_category=marks_category_constraints)
            for fact in identity_facts
        ],
        "全局共享高置信事实": [
            prompt_fact(fact, mark_negative=marks_negative_constraints)
            for fact in selected_facts
        ],
    }
    if prompt_version == "rag_v6":
        prompt_context["只读否定约束（不占Top-K，不要求写入文案）"] = [
            {
                **prompt_fact(fact),
                "约束": "只用于禁止相反内容；不得主动扩写为卖点",
            }
            for fact in negative_constraint_facts
        ]
    style_guidance = (
        RAG_STYLE_GUIDANCE[generation_input["category_l1"]]
        if prompt_version in {"rag_v3", "rag_v4", "rag_v5", "rag_v6"}
        else CATEGORY_STYLE_GUIDANCE[generation_input["category_l1"]]
    )
    content_requirements = CONTENT_REQUIREMENTS.copy()
    if prompt_version == "rag_v1":
        content_requirements["核心卖点"] = (
            "恰好生成三个核心卖点，每个卖点是一句话；可以从不同角度组织同一已知事实，"
            "但不得为了凑满三条而添加上下文没有提供的功能、效果或场景。"
        )
    else:
        content_requirements = {
            "标题": "只组合必要身份信息和高置信事实的原值，不添加评价性或效果性修饰。",
            "核心卖点": (
                "恰好生成三个字符串；每条只直接改写一个已提供的字段和值。事实不足时，"
                "允许使用品类、品牌或型号作为独立信息点，不能推导新的优势或用途。"
            ),
            "短详情": "只串联上下文中的字段和值，不解释这些事实会带来什么效果或好处。",
        }
    task_requirements = "\n".join(
        f"- {content_type}：{requirement}"
        for content_type, requirement in content_requirements.items()
    )
    system_prompt = (
        "你是专业的中文电商文案运营师。只能陈述用户提供的必要身份信息和高置信事实；"
        "不得依据常识、标题联想或营销习惯补充任何未提供的参数、功能、材质、效果、"
        "适用对象、场景、认证、促销、销量或售后承诺。事实不足时应保持概括，不能猜测。"
        "只输出合法 JSON，不要输出 Markdown 和解释。selling_points 数组必须恰好有三个字符串。"
    )
    if prompt_version == "rag_v2":
        system_prompt += (
            "品牌和型号仅用于标识商品，禁止从品牌名或型号文本拆解、推断连接方式、容量、"
            "适配设备或其他属性。禁止把属性解释成耐用、舒适、稳定、高效、安全、健康、"
            "便携、省力、易清洁、提升体验或适合某场景，除非上下文原值明确包含该结论。"
            "每个事实性短语都必须能逐字对应到上下文中的某个字段和值。"
        )
    if prompt_version in {"rag_v3", "rag_v4", "rag_v5"}:
        system_prompt += (
            "品牌和型号只用于原样标识商品，一级品类只用于控制表达风格，二级品类只用于商品命名。"
            "输出必须采用字段直述方式。除了为、是、采用、使用、配备、具有、容量、材质、接口、"
            "连接方式、型号、品牌这类连接词，不得加入上下文中没有出现的形容词、效果词、用途词或场景词。"
            "禁止写大容量、耐用、稳定、快速、高效、舒适、便捷、安全、健康、坚固、精准、"
            "适用、适合、满足、提升、节省、易于、带来、确保、保证等推断，除非该词就是上下文原值。"
            "三个卖点优先逐条写成“字段：值”；短详情只把身份信息和字段值连接成一句话。"
        )
    if prompt_version in {"rag_v4", "rag_v5"}:
        system_prompt += (
            "一级品类和二级品类都只是分类标签：可以原样写出，但绝不能改写成“适用于”“适合”"
            "“使用场景”或其他用途描述，除非高置信事实中另有明确场景字段。"
            "标记为否定事实的字段是硬约束，可以不写进卖点，但标题、卖点和短详情都不得生成与其相反的内容。"
            "selling_points 仍须恰好三个字符串：优先逐条使用非否定的高置信事实；不足三条时，"
            "只能用通过质量检查的品类、品牌或型号作为客观信息点补足，禁止添加效果、用途或场景。"
        )
    if prompt_version == "rag_v6":
        system_prompt += (
            "必要身份信息用于识别商品；全局共享高置信事实是唯一可以主动写入文案的内容事实；"
            "只读否定约束不占Top-K，只用于禁止生成相反内容，不要求写入标题、卖点或短详情。"
            "品类可以做不增加新事实的直接语义释义，例如只说明该类商品的基本用途；"
            "不得进一步推导上下文没有提供的具体场所、人群、对象、效果、性能或相对评价。"
            "selling_points 必须恰好三个字符串：内容事实不足时，只能用通过质量检查的品类、"
            "品牌或型号作为客观信息点补足。输出前检查每个事实性短语是否来自上述三类上下文。"
        )
    final_check = ""
    if prompt_version == "rag_v5":
        final_check = (
            "\n输出前必须静默检查并删除无事实来源的“适用于”“适合”“使用场景”等短语。"
            "例如，只有分类标签“家居日用”时，错误写法是“适用于家居日用”或“适用于家居收纳”，"
            "正确写法只能是“一级品类：家居日用”或直接省略一级品类。"
            "只有高置信事实中存在明确的适用场景、适用空间或适用对象字段时，才允许原样表达该字段值。"
        )
    user_prompt = (
        "请一次生成商品标题、核心卖点和短详情。三个内容类型分别遵循以下模板要求：\n"
        f"{task_requirements}\n"
        f"品类风格：{style_guidance}\n"
        "以下 RAG 上下文是本次唯一允许使用的商品事实。必要身份信息不计入 Top-K；"
        "高置信事实由三个生成任务联合排序后共享，所有内容都必须能由该上下文直接支持。\n"
        "输出 JSON 字段必须为 generated_title、selling_points、short_description，"
        "其中 selling_points 必须是包含三个字符串的数组。\n"
        f"{final_check}"
        f"RAG Prompt 版本：{prompt_version}\n"
        f"RAG 上下文：\n{json.dumps(prompt_context, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_generation_output(text: str) -> dict[str, Any]:
    """从模型回复中提取并校验约定的 JSON 文案。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace < 0 or last_brace < first_brace:
        raise ValueError("模型输出中没有完整 JSON 对象。")
    result = json.loads(stripped[first_brace : last_brace + 1])
    missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"模型输出缺少字段：{missing}")
    if not isinstance(result["generated_title"], str) or not result["generated_title"].strip():
        raise ValueError("generated_title 必须是非空字符串。")
    points = result["selling_points"]
    if not isinstance(points, list) or len(points) != 3 or not all(
        isinstance(point, str) and point.strip() for point in points
    ):
        raise ValueError("selling_points 必须是包含三个非空字符串的数组。")
    if not isinstance(result["short_description"], str) or not result["short_description"].strip():
        raise ValueError("short_description 必须是非空字符串。")
    return {field: result[field] for field in REQUIRED_OUTPUT_FIELDS}


class QwenGenerator:
    """使用 Transformers 和 bitsandbytes 运行 Qwen2.5-7B-Instruct。"""

    def __init__(self, config: dict[str, Any], cache_dir: Path, local_files_only: bool = False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "缺少生成模型依赖，请在 ecommerce-generation 环境安装 requirements-generation.txt。"
            ) from error

        if config["device"] != "cuda":
            raise ValueError("当前8GB显存验证只支持配置 device=cuda。")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 没有检测到可用的 NVIDIA CUDA 显卡。")

        cache_dir.mkdir(parents=True, exist_ok=True)
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(config["load_in_4bit"]),
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        common_options = {
            "cache_dir": str(cache_dir),
            "local_files_only": local_files_only,
        }
        if config.get("revision"):
            common_options["revision"] = str(config["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], **common_options)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map={"": 0},
            quantization_config=quantization_config,
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
            **common_options,
        )
        self.config = config
        self.cache_dir = cache_dir
        self.torch = torch
        self.torch.manual_seed(int(config["seed"]))
        self.torch.cuda.manual_seed_all(int(config["seed"]))

    def generate(
        self,
        generation_input: dict[str, Any],
        *,
        prompt_version: str | None = None,
        rag_context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        """生成一条文案，并且只返回新生成的文本。"""
        active_prompt_version = prompt_version or self.config["prompt_version"]
        if active_prompt_version in {
            "rag_v1",
            "rag_v2",
            "rag_v3",
            "rag_v4",
            "rag_v5",
            "rag_v6",
        }:
            if rag_context is None:
                raise ValueError("RAG Prompt 必须提供 rag_context。")
            messages = build_rag_messages(
                generation_input, rag_context, prompt_version=active_prompt_version
            )
        else:
            if rag_context is not None:
                raise ValueError("非 RAG Prompt 不能传入 rag_context。")
            messages = build_messages(generation_input, prompt_version=active_prompt_version)
        if seed is not None:
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = int(model_inputs["input_ids"].shape[-1])
        if input_length > int(self.config["max_input_tokens"]):
            raise ValueError(
                f"输入长度 {input_length} tokens 超过配置上限 {self.config['max_input_tokens']}。"
            )

        generation_options = {
            "max_new_tokens": int(self.config["max_new_tokens"]),
            "do_sample": bool(self.config["do_sample"]),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if generation_options["do_sample"]:
            generation_options.update(
                temperature=float(self.config["temperature"]),
                top_p=float(self.config["top_p"]),
            )
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_options)
        new_tokens = generated_ids[0, input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def model_identity(self) -> dict[str, str | None]:
        """返回本次实际加载的模型 revision 与本地 snapshot。"""
        revision = getattr(self.model.config, "_commit_hash", None)
        if revision is None:
            revision = self.tokenizer.init_kwargs.get("_commit_hash")
        configured_revision = self.config.get("revision")
        if revision is None and configured_revision:
            revision = str(configured_revision)
        snapshot_path = None
        if revision:
            repository_cache = "models--" + self.config["model_name"].replace("/", "--")
            candidate = self.cache_dir / repository_cache / "snapshots" / revision
            if candidate.is_dir():
                snapshot_path = str(candidate.resolve())
        return {
            "model_name": self.config["model_name"],
            "revision": revision,
            "snapshot_path": snapshot_path,
        }

    def gpu_memory_mib(self) -> dict[str, float]:
        """返回当前和峰值 PyTorch 显存，便于判断8GB显卡是否可用。"""
        return {
            "allocated": round(self.torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved": round(self.torch.cuda.memory_reserved() / 1024**2, 1),
            "peak_allocated": round(self.torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
