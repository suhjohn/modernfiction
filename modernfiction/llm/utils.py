# llm/utils.py
import base64
import os
import re
from typing import Any, Dict, List
from PIL import Image
import io


"""
https://docs.together.ai/docs/vision-overview

"""
provider_to_endpoint_map = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "together": "https://api.together.xyz/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "deepseek": "https://api.deepseek.com/chat/completions",
}


def get_endpoint(provider: str, model: str | None = None):
    if provider not in provider_to_endpoint_map:
        # check if provider is url
        if provider.startswith("http"):
            return provider
        raise ValueError(f"Invalid provider: {provider}")
    endpoint = provider_to_endpoint_map[provider]
    if provider == "gemini":
        if not model:
            raise ValueError("Model is required for Gemini")
        endpoint = endpoint.format(model=model)
        endpoint += f"?key={os.environ['GEMINI_API_KEY']}"
    return endpoint


def get_default_headers(provider: str):
    """Get default headers for different providers"""
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY'] or ''}"
    elif provider == "anthropic":
        headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"] or ""
        headers["anthropic-version"] = "2023-06-01"
    elif provider == "gemini":
        headers["X-Goog-Api-Key"] = f"Bearer {os.environ['GEMINI_API_KEY'] or ''}"
    elif provider == "deepseek":
        headers["Authorization"] = f"Bearer {os.environ['DEEPSEEK_API_KEY'] or ''}"
    else:
        headers["Authorization"] = f"Bearer {os.environ['TOGETHER_API_KEY'] or ''}"

    return headers


def image_to_base64(image: Image.Image):
    """
    Convert a PIL Image to base64 string.

    Args:
        image (Image.Image): PIL Image object

    Returns:
        str: Base64 encoded string of the image
    """
    # Convert PIL Image to bytes using context manager
    with io.BytesIO() as buffered:
        image.save(buffered, format="PNG")
        # Convert bytes to base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def openai_content_parser(response: Dict[str, Any]):
    """Parse the response from the LLM"""
    try:
        if response.get("choices"):
            content = response["choices"][0]["message"]["content"]
            return content
    except Exception:
        raise ValueError("No content found in the response", response)


def anthropic_content_parser(response: Dict[str, Any]):
    """Parse the response from the LLM"""
    try:
        if response.get("content"):
            content = response["content"][0]["text"]
            return content
    except Exception:
        raise ValueError("No content found in the response", response)


def gemini_content_parser(response: Dict[str, Any]):
    try:
        """Parse the response from the LLM"""
        if response.get("candidates"):
            content = response["candidates"][0]["content"]["parts"][0]["text"]
            return content
    except Exception:
        raise ValueError("No content found in the response", response)


def extract_tag(content: str, tag: str):
    match = re.search(f"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_tags(content: str, tag: str) -> List[str]:
    """
    Extract all occurrences of content within specified tags.

    Args:
        content (str): The input string containing tagged content
        tag (str): The tag to search for (e.g., 'content')

    Returns:
        List[str]: List of strings found between the specified tags
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    return [match.strip() for match in matches]
