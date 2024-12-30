# main.py
import asyncio
import logging
from typing import Awaitable, Callable
import uuid
import aiohttp
import bs4
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from retry import retry
import tqdm

from llm.transform_content import transform_content
from llm.should_transform import should_transform
from modernfiction.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


async def group_navigable_strings(
    p_tags: list[bs4.element.Tag],
) -> list[list[bs4.element.Tag]]:
    """
    Groups a list of <p> tags into sublists based on
    simple sentence boundary checks. A boundary is assumed if the
    last non-whitespace character of a paragraph is one of '.', '?', or '!'.
    """
    groups = []
    current_group = []

    for p_tag in p_tags:
        text = p_tag.get_text(strip=True)

        # Add the current <p> tag to the ongoing group
        current_group.append(p_tag)

        # If the last character indicates a sentence boundary, start a new group
        if text and text[-1] in [".", "?", "!"]:
            groups.append(current_group)
            current_group = []

    # Any leftover <p> tags go into a final group
    if current_group:
        groups.append(current_group)

    return groups


rate_limiter = RateLimiter(calls_per_minute=4000, max_parallel=100)


async def transform_text(
    content: str,
    provider: str,
    model: str,
    on_update_progress: Callable[[int], Awaitable[None]],
    progress_lock: asyncio.Lock | None = None,
) -> str:
    """
    Transform all <p> tags in `content`. Each time a chunk is done,
    we call pbar.update() (inside a lock) if pbar is provided.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1000),
        connector=aiohttp.TCPConnector(
            limit=100,
            ssl=False,
        ),
    ) as session:
        soup = BeautifulSoup(content, "html.parser")
        p_tags = soup.find_all("p")

        # Get batches of <p> tags
        batches = await group_navigable_strings(
            p_tags=p_tags,
        )

        @retry()
        async def process_paragraph_group(paragraphs: list[bs4.element.Tag]):
            # Our rate limiter allows concurrency, but you can also limit concurrency by
            # grouping paragraph calls if needed.
            async with rate_limiter:
                _should_transform = await should_transform(
                    session=session,
                    provider=provider,
                    model=model,
                    paragraphs=paragraphs,
                )
                if _should_transform:
                    try:
                        await transform_content(
                            session=session,
                            provider=provider,
                            model=model,
                            paragraphs=paragraphs,
                        )
                    except Exception:
                        await transform_content(
                            session=session,
                            provider="together",
                            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                            paragraphs=paragraphs,
                        )

        for batch in batches:
            # Process one batch at a time
            await process_paragraph_group(batch)
            # Update the progress bar if provided
            if on_update_progress is not None and progress_lock is not None:
                # Multiple tasks are updating the pbar, so protect it with a lock
                async with progress_lock:
                    await on_update_progress(len(batch))

    return str(soup)


async def transform_item(
    item: epub.EpubHtml,
    content: str,
    provider: str,
    model: str,
    on_update_progress: Callable[[int], Awaitable[None]],
    progress_lock: asyncio.Lock,
) -> epub.EpubHtml:
    """
    Helper coroutine to transform a single EpubHtml item and return a new EpubHtml.
    """
    new_content = await transform_text(
        content,
        provider=provider,
        model=model,
        on_update_progress=on_update_progress,
        progress_lock=progress_lock,  # pass the lock down
    )
    new_item = epub.EpubHtml(
        uid=item.id,
        file_name=item.file_name,
        media_type="application/xhtml+xml",
        content=new_content.encode("utf-8"),
    )
    new_item.is_chapter = item.is_chapter
    new_item.properties = item.properties

    # If there's any embedded <svg>, add the "svg" property
    if "<svg" in new_content.lower():
        if "svg" not in new_item.properties:
            new_item.properties.append("svg")

    return new_item


async def transform_epub(
    input_path: str,
    provider: str,
    model: str,
    on_init_progress: Callable[[int], Awaitable[None]],
    on_update_progress: Callable[[int], Awaitable[None]],
    on_complete_progress: Callable[[], Awaitable[None]],
) -> epub.EpubBook:
    # Read original EPUB
    book = epub.read_epub(input_path)

    # Create new book with proper initialization
    new_book = epub.EpubBook()
    new_book.set_identifier(str(uuid.uuid4()))

    # Copy over minimal metadata
    if book.metadata:
        for dc_key in ["title", "language", "creator", "identifier"]:
            values = book.get_metadata("DC", dc_key)
            if values:
                if dc_key == "identifier":
                    new_book.set_identifier(values[0][0])
                elif dc_key == "language":
                    new_book.set_language(values[0][0])
                elif dc_key == "title":
                    new_book.set_title(values[0][0])
                elif dc_key == "creator":
                    new_book.add_author(values[0][0])

    # -------------------------------------------------------------------
    # 1) Gather total paragraph count (preprocessing pass)
    # -------------------------------------------------------------------
    total_paragraphs = 0
    items_to_transform = []  # Keep track of (item, content) for second pass

    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            try:
                content = item.get_content()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                soup = BeautifulSoup(content, "html.parser")
                p_tags = soup.find_all("p")
                total_paragraphs += len(p_tags)
                # Save so we don’t re-parse the file again
                items_to_transform.append((item, content))
            except Exception as e:
                logger.error(f"Error reading {item.get_id()}: {e}")
        else:
            # Non-HTML items can be copied directly
            new_book.add_item(item)

    # -------------------------------------------------------------------
    # 2) Create a tqdm progress bar
    # -------------------------------------------------------------------
    if on_init_progress:
        await on_init_progress(total_paragraphs)

    # We need a lock so multiple tasks can safely update the bar
    progress_lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # 3) Transform each item **in parallel** using asyncio.gather
    # -------------------------------------------------------------------
    tasks = []
    for item, content in items_to_transform:
        tasks.append(
            transform_item(
                item, content, provider, model, on_update_progress, progress_lock
            )
        )

    # Wait for all tasks to complete
    new_items = await asyncio.gather(*tasks)

    # Close the progress bar
    if on_complete_progress:
        await on_complete_progress()

    # Add the newly transformed EpubHtml items
    for new_item in new_items:
        new_book.add_item(new_item)

    # Add a nav document if needed
    nav_doc = epub.EpubNav(uid="nav", file_name="nav.xhtml")
    new_book.add_item(nav_doc)

    # Rebuild the spine
    new_book.spine = [
        item.id for item in new_book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    ]

    # Copy the TOC from the original
    new_book.toc = book.toc

    return new_book
