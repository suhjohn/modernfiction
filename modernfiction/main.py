import asyncio
from dotenv import load_dotenv
from ebooklib import epub
import tqdm

from modernfiction.transform_text import transform_epub

load_dotenv()


def get_output_path(input_path: str, output_path: str | None = None) -> str:
    """Generate output path for transformed epub file."""
    if output_path:
        return output_path
    return input_path.replace(".epub", "_transformed.epub")


async def main(
    input_path: str,
    output_path: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
):
    pbar = tqdm.tqdm(total=0, desc="Transforming paragraphs")

    async def on_init_progress(total_paragraphs: int):
        pbar.total = total_paragraphs
        pbar.reset()

    async def on_update_progress(paragraphs_transformed: int):
        pbar.update(paragraphs_transformed)

    async def on_complete_progress():
        pbar.close()

    new_book = await transform_epub(
        input_path=input_path,
        provider=provider,
        model=model,
        on_init_progress=on_init_progress,
        on_update_progress=on_update_progress,
        on_complete_progress=on_complete_progress,
    )
    # Write the new epub
    print("Writing new epub...")
    epub.write_epub(output_path, new_book)
    print(f"Transformed EPUB written to {output_path}")


if __name__ == "__main__":
    import argparse
    import dotenv

    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Transform EPUB content.")
    parser.add_argument(
        "-i", "--input", help="Input EPUB file path", default="./gonewiththewind.epub"
    )
    parser.add_argument("-o", "--output", help="Output EPUB file path", default=None)
    parser.add_argument("-p", "--provider", help="LLM provider", default="openai")
    parser.add_argument("-m", "--model", help="LLM model", default="gpt-4o-mini")

    args = parser.parse_args()
    output_path = get_output_path(args.input, args.output)

    asyncio.run(main(args.input, output_path, args.provider, args.model))
