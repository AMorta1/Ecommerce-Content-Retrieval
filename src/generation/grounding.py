"""对 RAG 生成结果做确定性、只读的事实边界检查。"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable


VALIDATOR_VERSION = "rag_grounding_validator_v1"
ERROR_CLASSES = {
    "unsupported_usage_scenario",
    "unsupported_evaluative_claim",
    "grounding_failure",
}

CATEGORY_DIRECT_USE = {
    "收纳箱": {"收纳", "物品收纳", "收纳物品", "家居收纳", "家居日用收纳"},
    "拖把": {
        "地面清洁",
        "清洁地面",
        "日常清洁",
        "日常清洁使用",
        "家居清洁",
        "家居日常清洁",
        "家居日常清洁使用",
    },
    "垃圾桶": {"垃圾收纳", "收纳垃圾"},
}

USAGE_PATTERN = re.compile(r"(?:适用于|适用|适合|用于|用来)([^，。；;！？!?]+)")
EVALUATIVE_PATTERNS = (
    ("relative_capacity", re.compile(r"大容量")),
    ("subjective_feel", re.compile(r"(?:良好|出色|舒适|顺滑)(?:的)?手感")),
    (
        "stability_claim",
        re.compile(
            r"(?:稳定(?:的)?(?:输入|连接|传输|性能|体验|操控|输出)|"
            r"(?:输入|连接|传输|性能|体验|操控|输出)(?:更加|更)?稳定)"
        ),
    ),
    (
        "generic_evaluative_claim",
        re.compile(
            r"(?:高效|精准|舒适|便捷|方便|安全|健康|坚固|耐用|省力|易清洁|可靠|优质)"
        ),
    ),
)
EXPLICIT_USAGE_FIELDS = (
    "适用",
    "场景",
    "空间",
    "对象",
    "人群",
    "兼容",
    "用途",
    "佩戴",
)


def _normalized(text: str) -> str:
    return re.sub(r"[\s，。；;：:、,.!?！？()（）\[\]【】]", "", text).lower()


def _facts(rag_context: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    seen = set()
    for key in ("identity_facts", "selected_facts", "negative_constraint_facts"):
        for fact in rag_context.get(key, []):
            fact_id = fact.get("fact_id") or (
                fact.get("canonical_field"),
                tuple(fact.get("normalized_values", [])),
            )
            if fact_id not in seen:
                facts.append(fact)
                seen.add(fact_id)
    return facts


def _is_negative_fact(fact: dict[str, Any]) -> bool:
    values = [str(value).strip() for value in fact.get("normalized_values", [])]
    return any(
        value in {"无", "否", "不支持", "有线", "没有", "未配备"}
        or "不支持" in value
        for value in values
    )


def _output_parts(output: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "generated_title", str(output["generated_title"])
    for index, point in enumerate(output["selling_points"]):
        yield f"selling_points[{index}]", str(point)
    yield "short_description", str(output["short_description"])


def _usage_is_supported(
    target: str,
    *,
    category_l1: str,
    category_l2: str,
    facts: list[dict[str, Any]],
) -> tuple[bool, str]:
    normalized_target = _normalized(target)
    if normalized_target in {_normalized(category_l1), _normalized(category_l2)}:
        return True, "direct_category_label"
    if normalized_target in {
        _normalized(category_l1 + category_l2),
        _normalized(category_l2 + category_l1),
    }:
        return True, "direct_category_label_combination"
    if normalized_target in {
        _normalized(value) for value in CATEGORY_DIRECT_USE.get(category_l2, set())
    }:
        return True, "direct_category_semantics"

    for fact in facts:
        field = str(fact.get("canonical_field", ""))
        if not any(marker in field for marker in EXPLICIT_USAGE_FIELDS):
            continue
        for value in fact.get("normalized_values", []):
            normalized_value = _normalized(str(value))
            if normalized_value and (
                normalized_value in normalized_target or normalized_target in normalized_value
            ):
                return True, f"explicit_usage_fact:{field}"
            value_stem = normalized_value.removesuffix("式")
            if len(value_stem) >= 2 and value_stem in normalized_target:
                return True, f"explicit_usage_fact_paraphrase:{field}"
    return False, "no_category_or_explicit_usage_support"


def _claim_has_source_support(span: str, facts: list[dict[str, Any]]) -> bool:
    normalized_span = _normalized(span)
    values = [
        _normalized(str(value))
        for fact in facts
        for value in fact.get("normalized_values", [])
    ]
    if any(normalized_span and normalized_span in value for value in values):
        return True
    if "快速充电" in span and any("快充" in value for value in values):
        return True
    return False


def _negative_contradiction(
    output_text: str, fact: dict[str, Any]
) -> tuple[str, str] | None:
    field = str(fact.get("canonical_field", ""))
    values = [str(value).strip() for value in fact.get("normalized_values", [])]
    compact = _normalized(output_text)

    if field == "是否无线" and "有线" in values:
        cleaned = re.sub(r"是否无线(?:为)?有线|不是无线|非无线|有线", "", compact)
        if "无线" in cleaned:
            return "无线", "negative_constraint:是否无线=有线"
        return None

    negative_markers = ("无", "否", "不支持", "没有", "未配备", "非", "不是")
    if not any(value in {"无", "否", "不支持", "没有", "未配备"} or "不支持" in value for value in values):
        return None

    if field.startswith("是否有"):
        subject = field[3:]
    elif field.startswith("是否支持"):
        subject = field[4:]
    elif field.startswith("是否带"):
        subject = field[3:]
    elif field.startswith("是否"):
        subject = field[2:]
    else:
        return None
    subject = _normalized(subject)
    if not subject or subject not in compact:
        return None

    clauses = re.split(r"[，。；;、,.!?！？]", output_text)
    for clause in clauses:
        normalized_clause = _normalized(clause)
        if subject not in normalized_clause:
            continue
        has_negative_marker = any(marker + subject in normalized_clause for marker in negative_markers)
        has_field_negative_form = _normalized(field) in normalized_clause and any(
            _normalized(value) in normalized_clause for value in values
        )
        if not has_negative_marker and not has_field_negative_form:
            return subject, f"negative_constraint:{field}={'/'.join(values)}"
    return None


def _directional_fact_contradictions(
    output_text: str, facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    findings = []
    opposites = (("以下", "以上"), ("以上", "以下"), ("不超过", "超过"))
    compact_output = _normalized(output_text)
    for fact in facts:
        if fact.get("fact_role") == "identity":
            continue
        for raw_value in fact.get("normalized_values", []):
            value = str(raw_value).strip()
            for expected, opposite in opposites:
                if expected not in value:
                    continue
                prefix = _normalized(value.split(expected, 1)[0])
                if prefix and prefix + opposite in compact_output:
                    findings.append(
                        {
                            "classification": "grounding_failure",
                            "location": "all_output",
                            "text_span": prefix + opposite,
                            "reason": (
                                f"selected_fact_direction_contradiction:"
                                f"{fact.get('canonical_field')}={value}"
                            ),
                            "fact_id": fact.get("fact_id"),
                            "potential_false_positive": False,
                        }
                    )
    return findings


def validate_generation_grounding(
    output: dict[str, Any] | None,
    rag_context: dict[str, Any],
) -> dict[str, Any]:
    """检测事实边界问题；不修改、删除或重新生成输出。"""
    if output is None:
        findings = [
            {
                "classification": "grounding_failure",
                "location": "output",
                "text_span": "",
                "reason": "unparseable_output_cannot_be_grounded",
                "potential_false_positive": False,
            }
        ]
        return _result(findings)

    facts = _facts(rag_context)
    category_values = {
        str(fact.get("canonical_field")): str(fact.get("normalized_values", [""])[0])
        for fact in facts
        if fact.get("canonical_field") in {"category_l1", "category_l2"}
        and fact.get("normalized_values")
    }
    category_l1 = category_values.get("category_l1", "")
    category_l2 = category_values.get("category_l2", "")
    findings: list[dict[str, Any]] = []

    output_text = "\n".join(text for _, text in _output_parts(output))
    for location, text in _output_parts(output):
        for match in USAGE_PATTERN.finditer(text):
            target = match.group(1).strip()
            supported, reason = _usage_is_supported(
                target,
                category_l1=category_l1,
                category_l2=category_l2,
                facts=facts,
            )
            findings.append(
                {
                    "classification": (
                        "supported_paraphrase" if supported else "unsupported_usage_scenario"
                    ),
                    "location": location,
                    "text_span": match.group(0),
                    "reason": reason,
                    "potential_false_positive": False,
                }
            )

        for claim_type, pattern in EVALUATIVE_PATTERNS:
            for match in pattern.finditer(text):
                span = match.group(0)
                if _claim_has_source_support(span, facts):
                    continue
                findings.append(
                    {
                        "classification": "unsupported_evaluative_claim",
                        "location": location,
                        "text_span": span,
                        "reason": f"no_explicit_source_support:{claim_type}",
                        "potential_false_positive": False,
                    }
                )

    negative_facts = [fact for fact in facts if _is_negative_fact(fact)]
    for fact in negative_facts:
        contradiction = _negative_contradiction(output_text, fact)
        if contradiction is None:
            continue
        span, reason = contradiction
        findings.append(
            {
                "classification": "grounding_failure",
                "location": "all_output",
                "text_span": span,
                "reason": reason,
                "fact_id": fact.get("fact_id"),
                "potential_false_positive": False,
            }
        )
    findings.extend(_directional_fact_contradictions(output_text, facts))

    unique = []
    seen = set()
    for finding in findings:
        key = (
            finding["classification"],
            finding["location"],
            finding["text_span"],
            finding["reason"],
        )
        if key not in seen:
            unique.append(finding)
            seen.add(key)
    return _result(unique)


def _result(findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(finding["classification"] for finding in findings)
    error_count = sum(counts[classification] for classification in ERROR_CLASSES)
    return {
        "validator_version": VALIDATOR_VERSION,
        "mode": "detect_and_record_only",
        "output_modified": False,
        "regeneration_triggered": False,
        "replaces_human_evaluation": False,
        "status": "FAIL" if error_count else "PASS",
        "error_count": error_count,
        "classification_counts": dict(sorted(counts.items())),
        "findings": findings,
    }
