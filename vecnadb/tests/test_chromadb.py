import os
import pathlib

import vecnadb
from vecnadb.shared.logging_utils import get_logger
from vecnadb.infrastructure.files.storage import get_storage_config
from vecnadb.modules.data.models import Data
from vecnadb.modules.users.methods import get_default_user
from vecnadb.modules.search.types import SearchType
from vecnadb.modules.search.operations import get_history

logger = get_logger()


async def test_local_file_deletion(data_text, file_location):
    from sqlalchemy import select
    import hashlib
    from vecnadb.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        # Get hash of data contents
        encoded_text = data_text.encode("utf-8")
        data_hash = hashlib.md5(encoded_text).hexdigest()
        # Get data entry from database based on hash contents
        data = (await session.scalars(select(Data).where(Data.content_hash == data_hash))).one()
        assert os.path.isfile(data.raw_data_location.replace("file://", "")), (
            f"Data location doesn't exist: {data.raw_data_location}"
        )
        # Test deletion of data along with local files created by cognee
        await engine.delete_data_entity(data.id)
        assert not os.path.exists(data.raw_data_location.replace("file://", "")), (
            f"Data location still exists after deletion: {data.raw_data_location}"
        )

    async with engine.get_async_session() as session:
        # Get data entry from database based on file path
        data = (
            await session.scalars(select(Data).where(Data.raw_data_location == file_location))
        ).one()
        assert os.path.isfile(data.raw_data_location.replace("file://", "")), (
            f"Data location doesn't exist: {data.raw_data_location}"
        )
        # Test local files not created by cognee won't get deleted
        await engine.delete_data_entity(data.id)
        assert os.path.exists(data.raw_data_location.replace("file://", "")), (
            f"Data location doesn't exists: {data.raw_data_location}"
        )


async def test_getting_of_documents(dataset_name_1):
    # Test getting of documents for search per dataset
    from vecnadb.modules.users.permissions.methods import get_document_ids_for_user

    user = await get_default_user()
    document_ids = await get_document_ids_for_user(user.id, [dataset_name_1])
    assert len(document_ids) == 1, (
        f"Number of expected documents doesn't match {len(document_ids)} != 1"
    )

    # Test getting of documents for search when no dataset is provided
    user = await get_default_user()
    document_ids = await get_document_ids_for_user(user.id)
    assert len(document_ids) == 2, (
        f"Number of expected documents doesn't match {len(document_ids)} != 2"
    )


async def test_vector_engine_search_none_limit():
    file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    file_path_nlp = os.path.join(
        pathlib.Path(__file__).parent,
        "test_data/Natural_language_processing.txt",
    )

    await vecnadb.prune.prune_data()
    await vecnadb.prune.prune_system(metadata=True)

    await vecnadb.add(file_path_quantum)

    await vecnadb.add(file_path_nlp)

    await vecnadb.cognify()

    query_text = "Tell me about Quantum computers"

    from vecnadb.infrastructure.databases.vector import get_vector_engine

    vector_engine = get_vector_engine()

    collection_name = "Entity_name"

    query_vector = (await vector_engine.embedding_engine.embed_text([query_text]))[0]

    result = await vector_engine.search(
        collection_name=collection_name, query_vector=query_vector, limit=None
    )

    # Check that we did not accidentally use any default value for limit
    # in vector search along the way (like 5, 10, or 15)
    assert len(result) > 15


async def main():
    vecnadb.config.set_vector_db_config(
        {
            "vector_db_url": "http://localhost:3002",
            "vector_db_key": "test-token",
            "vector_db_provider": "chromadb",
        }
    )

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_chromadb")
        ).resolve()
    )
    vecnadb.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_chromadb")
        ).resolve()
    )
    vecnadb.config.system_root_directory(cognee_directory_path)

    await vecnadb.prune.prune_data()
    await vecnadb.prune.prune_system(metadata=True)

    dataset_name_1 = "natural_language"
    dataset_name_2 = "quantum"

    explanation_file_path_nlp = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    await vecnadb.add([explanation_file_path_nlp], dataset_name_1)

    explanation_file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    await vecnadb.add([explanation_file_path_quantum], dataset_name_2)

    await vecnadb.cognify([dataset_name_2, dataset_name_1])

    from vecnadb.infrastructure.databases.vector import get_vector_engine

    await test_getting_of_documents(dataset_name_1)

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
        query_type=SearchType.CHUNKS, query_text=random_node_name, datasets=[dataset_name_2]
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    graph_completion = await vecnadb.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=random_node_name,
        datasets=[dataset_name_2],
    )
    assert len(graph_completion) != 0, "Completion result is empty."
    print("Completion result is:")
    print(graph_completion)

    search_results = await vecnadb.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\n\nExtracted summaries are:\n")
    for result in search_results:
        print(f"{result}\n")

    user = await get_default_user()
    history = await get_history(user.id)
    assert len(history) == 8, "Search history is not correct."

    await vecnadb.prune.prune_data()
    data_root_directory = get_storage_config()["data_root_directory"]
    assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

    await vecnadb.prune.prune_system(metadata=True)
    tables_in_database = await vector_engine.get_collection_names()
    assert len(tables_in_database) == 0, "ChromaDB database is not empty"

    await test_vector_engine_search_none_limit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
