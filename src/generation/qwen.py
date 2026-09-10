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


def build_messages(generation_input: dict[str, Any]) -> list[dict[str, str]]:
    """把标准化商品属性转换为 Qwen 的 system/user 消息。"""
    validate_generation_input(generation_input)
    product_facts = {
        "一级品类": generation_input["category_l1"],
        "二级品类": generation_input["category_l2"],
        "商品属性": generation_input["attributes"],
    }
    style_guidance = CATEGORY_STYLE_GUIDANCE[generation_input["category_l1"]]
    task_requirements = "\n".join(
        f"- {content_type}：{requirement}"
        for content_type, requirement in CONTENT_REQUIREMENTS.items()
    )
    system_prompt = (
        "你是专业的中文电商文案运营师。只能依据用户提供的商品事实写作，不得编造参数、"
        "功能、认证、促销、销量或售后承诺。只输出合法 JSON，不要输出 Markdown 和解释。"
    )
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

    def generate(self, generation_input: dict[str, Any]) -> str:
        """生成一条文案，并且只返回新生成的文本。"""
        model_inputs = self.tokenizer.apply_chat_template(
            build_messages(generation_input),
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

    def gpu_memory_mib(self) -> dict[str, float]:
        """返回当前和峰值 PyTorch 显存，便于判断8GB显卡是否可用。"""
        return {
            "allocated": round(self.torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved": round(self.torch.cuda.memory_reserved() / 1024**2, 1),
            "peak_allocated": round(self.torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
