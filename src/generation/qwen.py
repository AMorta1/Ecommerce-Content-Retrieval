"""Qwen 文案生成、输入约束与结构化输出解析。"""

from __future__ import annotations

import hashlib
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
RAG_V2_PROMPT_VERSIONS = {
    "rag_v2_context_v1",
    "rag_v2_coverage_v1",
    "rag_v2_controlled_factual_v1",
}


def mandatory_fact_lines(rag_context: dict[str, Any]) -> list[str]:
    """返回 v2 Prompt 中 Mandatory facts 的稳定、可逐项验真的序列化文本。"""
    lines = []
    for fact in rag_context.get("mandatory_core_facts", []):
        values = "、".join(str(value) for value in fact["normalized_values"])
        lines.append(f"[{fact['fact_id']}] {fact['canonical_field']}={values}")
    return lines


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
    supported_versions = {
        "rag_v1",
        "rag_v2",
        "rag_v3",
        "rag_v4",
        "rag_v5",
        "rag_v6",
        *RAG_V2_PROMPT_VERSIONS,
    }
    if prompt_version not in supported_versions:
        raise ValueError(f"不支持的 RAG Prompt 版本：{prompt_version}")
    is_four_zone_v2 = prompt_version in RAG_V2_PROMPT_VERSIONS
    identity_facts = rag_context.get("identity_facts")
    selected_facts = (
        rag_context.get("supplemental_facts")
        if is_four_zone_v2
        else rag_context.get("selected_facts")
    )
    mandatory_core_facts = rag_context.get("mandatory_core_facts", [])
    negative_constraint_facts = rag_context.get("negative_constraint_facts", [])
    if not isinstance(identity_facts, list) or not isinstance(selected_facts, list):
        raise TypeError("rag_context 必须包含 Identity 和内容事实数组。")
    if not isinstance(mandatory_core_facts, list) or not isinstance(
        negative_constraint_facts, list
    ):
        raise TypeError("rag_context 的 Mandatory 与 Negative facts 必须是数组。")
    if is_four_zone_v2 and "mandatory_core_facts" not in rag_context:
        raise ValueError("RAG v2 必须提供 mandatory_core_facts。")
    if prompt_version == "rag_v6" and "negative_constraint_facts" not in rag_context:
        raise ValueError("rag_v6 必须提供独立的 negative_constraint_facts。")
    if len(selected_facts) > int(rag_context.get("top_k", -1)):
        raise ValueError("RAG 注入事实数量超过配置的 Top-K。")
    all_facts = [
        *identity_facts,
        *mandatory_core_facts,
        *selected_facts,
        *negative_constraint_facts,
    ]
    if any(fact.get("quality_status") != "eligible" for fact in all_facts):
        raise ValueError("RAG Prompt 只能注入 quality_status=eligible 的事实。")
    if any(fact.get("product_id") != rag_context.get("product_id") for fact in all_facts):
        raise ValueError("RAG Prompt 中存在跨商品事实。")
    if prompt_version == "rag_v6":
        selected_ids = {fact.get("fact_id") for fact in selected_facts}
        negative_ids = {fact.get("fact_id") for fact in negative_constraint_facts}
        if selected_ids & negative_ids:
            raise ValueError("rag_v6 内容 Top-K 与只读否定约束不能重复。")
    if is_four_zone_v2:
        zone_ids = []
        for facts in (
            identity_facts,
            mandatory_core_facts,
            selected_facts,
            negative_constraint_facts,
        ):
            ids = [fact.get("fact_id") for fact in facts]
            if any(not fact_id for fact_id in ids) or len(ids) != len(set(ids)):
                raise ValueError("RAG v2 各事实必须有唯一 fact_id。")
            zone_ids.extend(ids)
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("RAG v2 的四个事实区不得重复同一 fact。")
        expected_ids = rag_context.get("mandatory_injection_expected_fact_ids")
        actual_ids = [fact["fact_id"] for fact in mandatory_core_facts]
        if expected_ids != actual_ids:
            raise ValueError("RAG v2 Mandatory 预期 ID 与实际注入列表不一致。")

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

    if is_four_zone_v2:
        prompt_context = {
            "Identity Facts（身份信息）": [
                prompt_fact(fact, mark_category=True) for fact in identity_facts
            ],
            "Mandatory Reliable Core Facts（必须在全文完整覆盖）": mandatory_fact_lines(
                rag_context
            ),
            "Supplemental Top-3（可选补充细节）": [
                prompt_fact(fact) for fact in selected_facts
            ],
            "Negative Constraints（只读，不要求主动写入）": [
                {
                    **prompt_fact(fact),
                    "约束": "只用于禁止相反内容；不得主动扩写为卖点",
                }
                for fact in negative_constraint_facts
            ],
        }
        system_prompt = (
            "你是专业的中文电商文案运营师。只能使用本次 RAG 上下文中的商品事实，"
            "不得补充未提供的参数、功能、材质、效果、用途、场景、人群、认证、促销或承诺。"
            "Identity 只用于识别商品；Negative Constraints 只用于禁止相反内容；"
            "Supplemental 是可选细节。只输出合法 JSON，不要输出 Markdown 和解释。"
            "selling_points 必须恰好是三个非空字符串。品牌和型号只能原样标识商品，"
            "品类只能用于商品命名，禁止由品牌、型号或品类推断其他属性。"
            "输出必须采用中性字段直述方式，不得解释事实带来的好处或效果。"
            "除非词语本身就是上下文字段值，禁止写大容量、优质、耐用、稳定、快速、高效、"
            "精准、舒适、便捷、方便、安全、健康、坚固、轻松、易清洁、适用、适合、满足、"
            "提升、节省、带来、确保、保证等评价、效果、用途或场景表达。"
            "数值、范围、单位和否定方向必须保持上下文原值，不得换算、缩写或改写。"
        )
        coverage_requirement = ""
        if prompt_version == "rag_v2_coverage_v1":
            coverage_requirement = (
                "全文必须让每一条 Mandatory Reliable Core Fact 至少完整出现一次；"
                "不要求在标题、卖点和短详情中重复。标题优先承载品牌、型号、品类及关键识别属性，"
                "三条卖点承载主要参数、功能和材质，短详情补齐尚未表达的 Mandatory facts。"
                "输出前逐条核对 Mandatory 列表，不得遗漏或改写数值、单位与否定方向。"
            )
        elif prompt_version == "rag_v2_context_v1":
            coverage_requirement = (
                "Mandatory Reliable Core Facts 是可靠的核心内容事实，可在全文中自然选用；"
                "本版本不要求覆盖每一条，但所有已写事实仍必须逐字对应上下文。"
            )
        else:
            system_prompt = (
                "你是商品事实转写器，不是营销文案创作者。你的任务只是把本次 RAG 上下文"
                "已经提供的事实改写成指定 JSON，不得补充、推断或解释任何新事实。"
                "Identity 只用于原样识别商品；Mandatory Reliable Core Facts 是应优先表达的"
                "可靠核心事实；Supplemental 是可选细节；Negative Constraints 只用于禁止相反内容。"
                "只输出合法 JSON，不要输出 Markdown、分析过程或解释。selling_points 必须恰好"
                "包含三个非空字符串。"
                "品牌和型号只能逐字来自 Identity，缺失时宁可不写，禁止猜测、补全或替换。"
                "数值、单位、范围和边界必须保持源事实原义；禁止舍入、估算、换算、扩大、缩小"
                "或自行具体化。兼容设备、适用人群和适用范围必须保持输入中的具体对象与量词，"
                "禁止改写成各种设备、全部设备、广泛兼容、通用人群或其他扩大范围的说法。"
                "只能陈述事实是什么，禁止解释因此有什么好处。除非词语本身就是上下文字段值，"
                "禁止添加评价、效果、性能、体验、营销、用途或场景推导，包括优质、稳定、方便、"
                "便捷、耐用、安全、高效、精准、舒适、坚固、快速、环保、省心、出色、合理、"
                "确保、保证、提升、满足、带来、提供体验等表达。"
                "不得从品类推导具体场所、人群、兼容对象、性能或效果；只允许把品类原样作为"
                "商品名称，或作不新增具体事实的直接品类语义表达。"
            )
            coverage_requirement = (
                "完整文案应尽量让每条 Mandatory Reliable Core Fact 至少正确表达一次，但不得"
                "机械重复。若某条事实无法自然放入文案，可以不写；不得为了凑覆盖制造新事实、"
                "评价、效果、场景或扩大范围。"
            )
        user_prompt = (
            "请一次生成商品标题、三个核心卖点和短详情。\n"
            f"{coverage_requirement}\n"
            "标题只组合 Identity 与关键事实原值；三个卖点优先逐条写成“字段：值”；"
            "短详情只用“为、是、采用、使用、配备、具有、以及、其中、并”等少量中性连接词"
            "串联已有字段和值，不增加属性解释。"
            "事实不足时用合格的品类、品牌或型号作为客观信息点，不得用推断凑满。\n"
            "输出 JSON 字段必须为 generated_title、selling_points、short_description。\n"
            f"品类风格：{RAG_STYLE_GUIDANCE[generation_input['category_l1']]}\n"
            f"RAG Prompt 版本：{prompt_version}\n"
            f"RAG 上下文：\n{json.dumps(prompt_context, ensure_ascii=False, indent=2)}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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


def _token_id_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise TypeError("tokenizer 返回了无法识别的 input_ids。")
    return value


def _contains_token_subsequence(tokens: list[int], expected: list[int]) -> bool:
    if not expected:
        return False
    limit = len(tokens) - len(expected) + 1
    return any(tokens[index : index + len(expected)] == expected for index in range(limit))


def preflight_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_input_tokens: int,
    mandatory_lines: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """在禁用截断的前提下验证 Prompt 长度与 Mandatory token 完整性。"""
    mandatory_lines = list(mandatory_lines or [])
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    missing_text = [line for line in mandatory_lines if line not in rendered_prompt]
    if missing_text:
        raise ValueError(f"Mandatory facts 未完整渲染进 Prompt：{missing_text}")

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        truncation=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = _token_id_list(model_inputs["input_ids"])
    if len(input_ids) > max_input_tokens:
        raise ValueError(
            f"输入长度 {len(input_ids)} tokens 超过配置上限 {max_input_tokens}；"
            "禁止截断 Mandatory facts。"
        )

    missing_after_tokenization = []
    decoded_model_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    for line in mandatory_lines:
        encoded = tokenizer(
            line,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        expected_ids = _token_id_list(encoded["input_ids"])
        if not _contains_token_subsequence(input_ids, expected_ids) and line not in decoded_model_input:
            missing_after_tokenization.append(line)
    if missing_after_tokenization:
        raise ValueError(
            "Mandatory facts 在 tokenizer 后未完整进入模型上下文："
            f"{missing_after_tokenization}"
        )

    token_payload = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    metadata = {
        "truncation": False,
        "input_token_count": len(input_ids),
        "max_input_tokens": max_input_tokens,
        "input_token_ids_sha256": hashlib.sha256(token_payload).hexdigest(),
        "mandatory_expected_count": len(mandatory_lines),
        "mandatory_verified_count": len(mandatory_lines),
        "mandatory_injection_complete": True,
        "mandatory_injection_rate": 1.0 if mandatory_lines else None,
        "mandatory_verification": "token_subsequence_or_exact_decoded_text",
    }
    return model_inputs, metadata


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
        self.last_prompt_preflight: dict[str, Any] | None = None
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
            *RAG_V2_PROMPT_VERSIONS,
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
        mandatory_lines = (
            mandatory_fact_lines(rag_context)
            if active_prompt_version in RAG_V2_PROMPT_VERSIONS and rag_context is not None
            else []
        )
        model_inputs, self.last_prompt_preflight = preflight_chat_prompt(
            self.tokenizer,
            messages,
            max_input_tokens=int(self.config["max_input_tokens"]),
            mandatory_lines=mandatory_lines,
        )
        model_inputs = model_inputs.to(self.model.device)
        input_length = int(model_inputs["input_ids"].shape[-1])

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
