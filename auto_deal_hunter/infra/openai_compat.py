import json
from typing import TypeVar

from pydantic import BaseModel

from auto_deal_hunter.infra import usage
from auto_deal_hunter.infra.config import LLM_SEED, LLM_TEMPERATURE, OPENAI_API_STYLE

T = TypeVar("T", bound=BaseModel)


def _use_responses(client) -> bool:
    return OPENAI_API_STYLE == "responses" and hasattr(client, "responses")


def _responses_parsed(response):
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed
    raise ValueError("Responses API response did not contain parsed structured output")


def _responses_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return output_text
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text is not None:
                return text
    raise ValueError("Responses API response did not contain output text")


def parse_structured(
    client,
    *,
    model: str,
    text_format: type[T],
    user_prompt: str,
    instructions: str | None = None,
) -> T:
    if _use_responses(client):
        response = client.responses.parse(
            model=model,
            instructions=instructions,
            input=user_prompt,
            text_format=text_format,
            temperature=LLM_TEMPERATURE,
        )
        usage.TRACKER.record(model, getattr(response, "usage", None))
        return _responses_parsed(response)

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_prompt})
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=text_format,
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
        )
        usage.TRACKER.record(model, getattr(response, "usage", None))
        return response.choices[0].message.parsed
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
        )
        usage.TRACKER.record(model, getattr(response, "usage", None))
        data = json.loads(response.choices[0].message.content or "{}")
        return text_format(**data)


def generate_text(
    client,
    *,
    model: str,
    user_prompt: str,
    instructions: str | None = None,
) -> str:
    if _use_responses(client):
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_prompt,
            temperature=LLM_TEMPERATURE,
        )
        usage.TRACKER.record(model, getattr(response, "usage", None))
        return _responses_text(response)

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
    )
    usage.TRACKER.record(model, getattr(response, "usage", None))
    return response.choices[0].message.content
