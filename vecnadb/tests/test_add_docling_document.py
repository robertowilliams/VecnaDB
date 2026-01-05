import asyncio
import vecnadb

import os


async def main():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path_artificial = os.path.join(
        current_directory, "test_data", "artificial-intelligence.pdf"
    )
    file_path_png = os.path.join(current_directory, "test_data", "example_copy.png")
    file_path_pptx = os.path.join(current_directory, "test_data", "example.pptx")

    await vecnadb.prune.prune_data()
    await vecnadb.prune.prune_system(metadata=True)

    # Import necessary converter, and convert file to DoclingDocument format
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    result = converter.convert(file_path_artificial)
    await vecnadb.add(result.document)

    result = converter.convert(file_path_png)
    await vecnadb.add(result.document)

    result = converter.convert(file_path_pptx)
    await vecnadb.add(result.document)

    await vecnadb.cognify()

    answer = await vecnadb.search("Tell me about Artificial Intelligence.")
    assert len(answer) != 0

    answer = await vecnadb.search("Do programmers change light bulbs?")
    assert len(answer) != 0
    lowercase_answer = answer[0]["search_result"][0].lower()
    assert ("no" in lowercase_answer) or ("none" in lowercase_answer)

    answer = await vecnadb.search("What colours are there in the presentation table?")
    assert len(answer) != 0
    lowercase_answer = answer[0]["search_result"][0].lower()
    assert (
        ("red" in lowercase_answer)
        and ("blue" in lowercase_answer)
        and ("green" in lowercase_answer)
    )


if __name__ == "__main__":
    asyncio.run(main())
