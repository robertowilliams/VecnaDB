import os
import pathlib
import asyncio

import vecnadb
import vecnadb.modules.ingestion as ingestion
from vecnadb.infrastructure.llm import get_max_chunk_tokens
from vecnadb.infrastructure.llm.extraction import extract_content_graph
from vecnadb.modules.chunking.TextChunker import TextChunker
from vecnadb.modules.data.processing.document_types import TextDocument
from vecnadb.modules.users.methods import get_default_user
from vecnadb.shared.data_models import KnowledgeGraph
from vecnadb.tasks.documents import extract_chunks_from_documents
from vecnadb.tasks.ingestion import save_data_item_to_storage
from vecnadb.infrastructure.files.utils.open_data_file import open_data_file


async def extract_graphs(document_chunks):
    """
    Extract graph, and check if entities are present
    """

    extraction_results = await asyncio.gather(
        *[extract_content_graph(chunk.text, KnowledgeGraph) for chunk in document_chunks]
    )

    return all(
        any(
            term in node.name.lower()
            for extraction_result in extraction_results
            for node in extraction_result.nodes
        )
        for term in ("qubit", "algorithm", "superposition")
    )


async def main():
    """
    Test how well the entity extraction works. Repeat graph generation a few times.
    If 80% or more graphs are correctly generated, the test passes.
    """

    file_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent, "test_data/Quantum_computers.txt"
    )

    await vecnadb.prune.prune_data()
    await vecnadb.prune.prune_system(metadata=True)

    await vecnadb.add("NLP is a subfield of computer science.")

    original_file_path = await save_data_item_to_storage(file_path)

    async with open_data_file(original_file_path) as file:
        classified_data = ingestion.classify(file)

        # data_id is the hash of original file contents + owner id to avoid duplicate data
        data_id = await ingestion.identify(classified_data, await get_default_user())

    await vecnadb.add(file_path)

    text_document = TextDocument(
        id=data_id,
        type="text",
        mime_type="text/plain",
        name="quantum_text",
        raw_data_location=file_path,
        external_metadata=None,
    )

    document_chunks = []
    async for chunk in extract_chunks_from_documents(
        [text_document], max_chunk_size=get_max_chunk_tokens(), chunker=TextChunker
    ):
        document_chunks.append(chunk)

    number_of_reps = 5

    graph_results = await asyncio.gather(
        *[extract_graphs(document_chunks) for _ in range(number_of_reps)]
    )

    correct_graphs = [result for result in graph_results if result]

    assert len(correct_graphs) >= 0.8 * number_of_reps


if __name__ == "__main__":
    asyncio.run(main())
