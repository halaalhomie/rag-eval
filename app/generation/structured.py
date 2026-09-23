"""Structured (JSON) output on top of plain chat completion.

Many OpenAI-compatible servers (including mlx_lm.server) do not enforce JSON schemas, so
schema adherence is handled here and works identically across providers:

1. The Pydantic schema is described in the system prompt.
2. The reply is parsed tolerantly: code fences are stripped, and the first balanced
   JSON object is extracted from any surrounding prose.
3. Validation errors are fed back to the model for a bounded number of repair attempts.
4. If it still fails, `StructuredOutputError` is raised. Callers decide on a fallback;
   nothing silently substitutes a default value.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from app.generation.llm import ChatMessage, LLMClient, LLMResponse, StructuredOutputError


def extract_json_object(text: str) -> dict:
    """Return the first balanced top-level JSON object in `text`."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError("no JSON object found in reply")


def schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with ONLY a single JSON object, no prose and no code fences, "
        f"matching this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )


def complete_structured[T: BaseModel](
    client: LLMClient,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    model: str | None = None,
    max_tokens: int | None = 512,
    repair_attempts: int = 1,
) -> tuple[T, list[LLMResponse]]:
    """Returns the parsed object plus every raw response (for token and cost accounting)."""
    convo = [ChatMessage(role="system", content=schema_instruction(schema)), *messages]
    responses: list[LLMResponse] = []
    last_error = ""
    for _ in range(repair_attempts + 1):
        response = client.complete(convo, model=model, max_tokens=max_tokens, temperature=0.0)
        responses.append(response)
        try:
            return schema.model_validate(extract_json_object(response.text)), responses
        except (ValueError, ValidationError) as exc:
            last_error = str(exc).splitlines()[0][:300]
            convo = [
                *convo,
                ChatMessage(role="assistant", content=response.text),
                ChatMessage(
                    role="user",
                    content=f"That reply was invalid ({last_error}). "
                    "Reply again with only the corrected JSON object.",
                ),
            ]
    raise StructuredOutputError(
        f"{schema.__name__}: no valid output after {len(responses)} attempts: {last_error}"
    )
