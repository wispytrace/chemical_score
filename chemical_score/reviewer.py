"""Optional LLM narrative layer, intentionally separated from deterministic scores."""

from __future__ import annotations

import json
import os
from typing import Any


class QwenReviewer:
    """Generate a narrative report without changing deterministic engine scores."""

    def __init__(self, api_key: str | None = None, model: str = "qwen-plus") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "install the optional LLM dependencies with: pip install -e '.[llm]'"
            ) from exc
        token = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not token:
            raise ValueError("DASHSCOPE_API_KEY is required for QwenReviewer")
        self.model = model
        self.client = OpenAI(
            api_key=token,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_report(
        self,
        reactants_smiles: str,
        product_smiles: str,
        evaluation: dict[str, Any],
    ) -> str:
        prompt = (
            "你是计算化学与有机合成审阅者。请解释以下规则评分，明确指出它不是实验成功率，"
            "逐级引用维度和叶子指标证据，说明局限，并给出简洁建议。不要修改引擎分数。\n\n"
            f"反应物: {reactants_smiles}\n产物: {product_smiles}\n"
            f"评分 JSON:\n{json.dumps(evaluation, ensure_ascii=False, indent=2)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
