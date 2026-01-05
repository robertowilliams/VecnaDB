import os
import shutil
import vecnadb
import pathlib

from vecnadb.infrastructure.files.storage import get_storage_config
from vecnadb.modules.engine.models import NodeSet
from vecnadb.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from vecnadb.shared.logging_utils import get_logger
from vecnadb.modules.search.types import SearchType
from vecnadb.modules.search.operations import get_history
from vecnadb.modules.users.methods import get_default_user

logger = get_logger()


async def main():
    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_kuzu")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_kuzu")
        ).resolve()
    )

    try:
        # Set Kuzu as the graph database provider
        vecnadb.config.set_graph_database_provider("kuzu")
        vecnadb.config.data_root_directory(data_directory_path)
        vecnadb.config.system_root_directory(cognee_directory_path)

        await vecnadb.prune.prune_data()
        await vecnadb.prune.prune_system(metadata=True)

        dataset_name = "cs_explanations"

        explanation_file_path_nlp = os.path.join(
            pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
        )
        await vecnadb.add([explanation_file_path_nlp], dataset_name)

        explanation_file_path_quantum = os.path.join(
            pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
        )

        from vecnadb.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()

        is_empty = await graph_engine.is_empty()

        assert is_empty, "Kuzu graph database is not empty"

        await vecnadb.add([explanation_file_path_quantum], dataset_name)

        is_empty = await graph_engine.is_empty()

        assert is_empty, "Kuzu graph database should be empty before cognify"

        await vecnadb.cognify([dataset_name])

        is_empty = await graph_engine.is_empty()

        assert not is_empty, "Kuzu graph database should not be empty"

        from vecnadb.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        random_node = (await vector_engine.search("Entity_name", "Quantum computer"))[0]
        random_node_name = random_node.payload["text"]

        search_results = await vecnadb.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
        )
        assert len(search_results) != 0, "The search results list is empty."
        print("\n\nExtracted sentences are:\n")
        for result in search_results:
            print(f"{result}\n")

        search_results = await vecnadb.search(
            query_type=SearchType.CHUNKS, query_text=random_node_name
        )
        assert len(search_results) != 0, "The search results list is empty."
        print("\n\nExtracted chunks are:\n")
        for result in search_results:
            print(f"{result}\n")

        search_results = await vecnadb.search(
            query_type=SearchType.SUMMARIES, query_text=random_node_name
        )
        assert len(search_results) != 0, "Query related summaries don't exist."
        print("\nExtracted summaries are:\n")
        for result in search_results:
            print(f"{result}\n")

        user = await get_default_user()
        history = await get_history(user.id)
        assert len(history) == 6, "Search history is not correct."

        nodeset_text = "Neo4j is a graph database that supports cypher."

        await vecnadb.add([nodeset_text], dataset_name, node_set=["first"])

        await vecnadb.cognify([dataset_name])

        context_nonempty = await GraphCompletionRetriever(
            node_type=NodeSet,
            node_name=["first"],
        ).get_context("What is in the context?")

        context_empty = await GraphCompletionRetriever(
            node_type=NodeSet,
            node_name=["nonexistent"],
        ).get_context("What is in the context?")

        assert isinstance(context_nonempty, list) and context_nonempty != [], (
            f"Nodeset_search_test:Expected non-empty string for context_nonempty, got: {context_nonempty!r}"
        )

        assert context_empty == [], (
            f"Nodeset_search_test:Expected empty string for context_empty, got: {context_empty!r}"
        )

        await vecnadb.prune.prune_data()
        data_root_directory = get_storage_config()["data_root_directory"]
        assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

        await vecnadb.prune.prune_system(metadata=True)

        is_empty = await graph_engine.is_empty()

        assert is_empty, "Kuzu graph database is not empty"

    finally:
        # Ensure cleanup even if tests fail
        for path in [data_directory_path, cognee_directory_path]:
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
