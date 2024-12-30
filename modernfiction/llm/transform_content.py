# llm/transform_content.py
import logging
import os
from typing import Any, Dict, List
import aiohttp
import bs4
from retry import retry

from .utils import (
    anthropic_content_parser,
    extract_tags,
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
    additional_prompt: str | None = None,
):
    """Format messages according to provider specifications"""
    total_string_tags = text.count("<string>")
    prompt = f"""
 
Please rewrite the text into clear, modern English, while preserving the original meaning, tone, and style as much as possible. 
Change archaic words or phrases, and update the syntax so it flows naturally for a contemporary reader. 
Keep proper nouns and names as consistent as possible and keep historical or cultural references where they are integral to the text.
Keep numbers, dates, or other numerical values. 
Keep technical terms and acronyms as they are as well.
KEEP NUMBERED LISTS AND FORMATS. 
If it's not necessary to transform the text, just return it as is. 
Only respond with the transformed text. 

Write the transformed text in the same HTML format as the original text. 

Keep the same number of <string></string> tags as the original text.

Examples:
<input>
    <string>[Note: The introduction, notes and index have been omitted.]  </string>
</input>
<output>
    <string>[Note: The introduction, notes and index have been omitted.]  </string>
</output>
<input>
    <string><strong>Release date</strong>: December 1, 1999 [eBook #2017]<br/></string>
    <string>Most recently updated: December 11, 2024</string>
</input>
<output>
    <string><strong>Release date</strong>: December 1, 1999 [eBook #2017]<br/></string>
    <string>Most recently updated: December 11, 2024</string>
</output>
<input>
    <string>Sir, in this audience,</string>
    <string>Let my disclaiming from a purpos’d evil</string>
    <string>Free me so far in your most generous thoughts</string>
    <string>That I have shot my arrow o’er the house</string>
    <string>And hurt my brother.</string>
</input>
<output>
    <string>Sir, before everyone here,</string>
    <string>let my denial of any intended harm</string>
    <string>clear me in your generous opinion</string>
    <string>so that you understand I aimed too high</string>
    <string>and hurt my brother.</string>
</output>
<input>
    <string>What, art a heathen? How dost thou understand the Scripture? The Scripture says
    Adam digg’d. Could he dig without arms? I’ll put another question to thee. If
    thou answerest me not to the purpose, confess thyself—</string>
    <string>SECOND CLOWN.</string>
</input>
<output>
    <string>What? Are you a pagan? How do you understand the Scripture? The Scripture says Adam dug. Could he dig without arms? I’ll ask you another question. If you don't answer me properly, admit you're—</string>
    <string>SECOND CLOWN.</string>
</output>
<input>
    <string>]</string>
    <string><strong>Enter</strong></string>
    <string>Fortinbras, the English Ambassadors</string>
    <string>and
    others.</string>
</input>
<output>
    <string>]</string>
    <string><strong>Enter</strong></string>
    <string>Fortinbras, the English Ambassadors</string> 
    <string>and
    others.</string>
</output>

There are {total_string_tags} tags in the original text.
{additional_prompt or ""}
"""
    if provider == "anthropic":
        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": prompt})
        return {
            "model": model,
            "system": "Only respond with raw transformed text. Do not include any other text or formatting.",
            "messages": [
                {"role": "user", "content": content},
            ],
        }
    elif provider == "openai":
        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": prompt})
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
        messages.append({"role": "user", "content": text})
        messages.append({"role": "user", "content": prompt})
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
        content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": prompt})
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": content},
            ],
        }
    else:
        raise ValueError(f"Invalid provider: {provider}")


@retry(exceptions=Exception, tries=3)
async def transform_content(
    session: aiohttp.ClientSession,
    provider: str,
    model: str,
    paragraphs: List[bs4.element.Tag],
    attempt: int = 0,
    additional_prompt: str | None = None,
    endpoint: str | None = None,
    custom_headers: Dict[str, Any] | None = None,
    custom_payload: Dict[str, Any] | None = None,
):
    """
    Transform a list of <p> tags by sending them to an LLM.
    Replaces each paragraph's inner HTML with the transformed content.

    :param session: An aiohttp.ClientSession
    :param provider: LLM provider (e.g. "openai", "anthropic")
    :param model: Model name (e.g. "gpt-3.5-turbo")
    :param paragraphs: A list of BeautifulSoup <p> tags
    :param attempt: The current number of attempts (for retries)
    :param additional_prompt: Additional instructions appended if the transformation count doesn't match
    :param endpoint: Custom endpoint URL (if needed)
    :param custom_headers: Custom headers (if needed)
    :param custom_payload: Additional fields to merge into the payload (if needed)
    """

    # Determine endpoint/headers if none are passed
    endpoint = endpoint or get_endpoint(provider=provider, model=model)
    headers = custom_headers or get_default_headers(provider)

    # Replace the single-batch processing with a loop over chunks
    for start_index in range(0, len(paragraphs), CHUNK_SIZE):
        chunk = paragraphs[start_index : start_index + CHUNK_SIZE]

        # Build strings for this chunk only
        formatted_paragraphs = []
        meaningful_chunk_count = 0
        for p_tag in chunk:
            inner_html = p_tag.decode_contents().strip()
            if not inner_html:
                continue
            meaningful_chunk_count += 1
            formatted_paragraphs.append(f"<string>{inner_html}</string>")
        if not formatted_paragraphs:
            continue
        text = "\n".join(formatted_paragraphs)

        # Build the base payload (prompt)
        payload = get_payload(
            model=model,
            provider=provider,
            text=text,
            additional_prompt=additional_prompt,
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

        try:
            split_transformed_content = extract_tags(val, "string")

            # If the LLM returned the wrong number of paragraphs, retry (up to 3 tries)
            if (
                len(split_transformed_content) != len(formatted_paragraphs)
                and attempt < 3
            ):
                print(
                    f"========================================================================\n"
                    f"ERROR WHILE TRANSFORMING:\n"
                    f"------------------------------------------\n"
                    f"Original content:\n{text}\n\n"
                    f"------------------------------------------\n"
                    f"Transformed content: \n{val}\n\n"
                    f"------------------------------------------\n"
                    f"Retrying with additional prompt...\n"
                    f"========================================================================\n"
                )
                previous_additional_prompt = additional_prompt or ""
                return await transform_content(
                    session=session,
                    provider=provider,
                    model=model,
                    paragraphs=chunk,
                    attempt=attempt + 1,
                    additional_prompt=f"""
{previous_additional_prompt}
<attempt>
    <output>
        {val}
    </output>
    <reason>
        This is wrong because there are {meaningful_chunk_count} paragraphs in the original text (this chunk),
        but only {len(split_transformed_content)} strings in the transformed text.
        Please fix the transformed text so that it has the same number of paragraphs as the original text in the chunk.
    </reason>
</attempt>
""",
                )
            elif (
                len(split_transformed_content) != meaningful_chunk_count
                and attempt >= 3
            ):
                raise Exception(f"Failed to transform content after 3 attempts: {resp}")

            # Replace each paragraph's inner HTML with the transformed content

            i = 0
            for p_tag in chunk:
                inner_html = p_tag.decode_contents().strip()
                if not inner_html:
                    continue

                logger.info(
                    f"==========================================\n"
                    f"[FROM] {formatted_paragraphs[i]}\n"
                    f"[TO]   {split_transformed_content[i]}\n"
                    f"=========================================="
                )
                soup_fragment = bs4.BeautifulSoup(
                    split_transformed_content[i], "html.parser"
                )
                p_tag.clear()
                p_tag.extend(soup_fragment.contents)
                i += 1

        except Exception as e:
            logger.info(
                f"========================================================================\n"
                f"ERROR WHILE TRANSFORMING:\n"
                f"------------------------------------------\n"
                f"Original content:\n{text}\n\n"
                f"------------------------------------------\n"
                f"Transformed content: \n{val}\n\n"
                f"------------------------------------------\n"
                f"Error: \n{e}\n\n"
                f"========================================================================\n"
            )
            raise e
