# llm/transform_content.py
import logging
import os
from typing import Any, Dict, List
import aiohttp
import bs4
from retry import retry

from .utils import (
    anthropic_content_parser,
    extract_tag,
    gemini_content_parser,
    get_endpoint,
    get_default_headers,
    openai_content_parser,
)

START_OF_BOOK = "/Start of the book"

CHUNK_SIZE = 10
logger = logging.getLogger(__name__)


def get_payload(
    model: str,
    provider: str,
    text: str,
):
    """Format messages according to provider specifications"""
    prompt = """ 
You are a helpful assistant trying to determine if the text should be transformed into modern English.

If the text in the string tags are the following, return false:
- A list of dates
- A list of names
- A list of acronyms
- A list of technical terms
- A list of numbers

If the text in the string tags is ANY of the following, return true:
- A sentence from a book

Respond: 
<should_transform>
true|false
</should_transform>
"""
    if provider == "anthropic":
        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        content.append({"type": "text", "text": text})
        return {
            "model": model,
            "system": "Only respond with raw transformed text. Do not include any other text or formatting.",
            "messages": [
                {"role": "user", "content": content},
            ],
        }
    elif provider == "openai":
        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        content.append({"type": "text", "text": text})
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": content},
            ],
        }
    elif provider == "together":
        messages: List[Dict[str, Any]] = []
        messages.append(
            {
                "role": "system",
                "content": "Only respond with raw transformed text.",
            }
        )
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "user", "content": text})
        return {
            "model": model,
            "messages": messages,
        }
    elif provider == "gemini":
        parts: List[Dict[str, Any]] = []
        parts.append({"text": prompt})
        parts.append({"text": f"<input>{text}</input>"})
        return {
            "model": model,
            "contents": [{"parts": parts}],
        }
    elif provider == "deepseek":
        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        content.append({"type": "text", "text": text})
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": content},
            ],
        }
    else:
        raise ValueError(f"Invalid provider: {provider}")


@retry(exceptions=Exception, tries=3)
async def should_transform(
    session: aiohttp.ClientSession,
    provider: str,
    model: str,
    paragraphs: List[bs4.element.Tag],
    endpoint: str | None = None,
    custom_headers: Dict[str, Any] | None = None,
    custom_payload: Dict[str, Any] | None = None,
):
    # Determine endpoint/headers if none are passed
    endpoint = endpoint or get_endpoint(provider=provider, model=model)
    headers = custom_headers or get_default_headers(provider)

    # Replace the single-batch processing with a loop over chunks
    for start_index in range(0, len(paragraphs), CHUNK_SIZE):
        chunk = paragraphs[start_index : start_index + CHUNK_SIZE]

        # Build strings for this chunk only
        formatted_paragraphs = []
        for p_tag in chunk:
            inner_html = p_tag.decode_contents().strip()
            formatted_paragraphs.append(f"<string>{inner_html}</string>")

        text = "\n".join(formatted_paragraphs)

        # Build the base payload (prompt)
        payload = get_payload(
            model=model,
            provider=provider,
            text=text,
        )
        # Merge any custom payload entries
        if custom_payload is not None:
            payload.update(custom_payload)

        # Send POST request
        response = await session.post(endpoint, headers=headers, json=payload)
        resp = await response.json()

        # Parse out LLM response
        if provider == "openai" or provider == "deepseek":
            val = openai_content_parser(resp)
            if val is None:
                raise Exception(f"Failed to transform content: {resp}")
        elif provider == "gemini":
            val = gemini_content_parser(resp)
            if val is None:
                raise Exception(f"Failed to transform content: {resp}")
        elif provider == "anthropic":
            val = anthropic_content_parser(resp)
            if val is None:
                raise Exception(f"Failed to transform content: {resp}")
        elif provider == "together":
            val = openai_content_parser(resp)
            if val is None:
                raise Exception(f"Failed to transform content: {resp}")
        else:
            raise ValueError(f"Invalid provider: {provider}")

        should_transform = extract_tag(val, "should_transform")
        if should_transform == "true":
            return True
        elif should_transform == "false":
            return False
        return True
