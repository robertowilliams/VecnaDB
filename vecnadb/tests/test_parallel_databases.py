import os
import pathlib
import vecnadb
from vecnadb.modules.search.operations import get_history
from vecnadb.modules.users.methods import get_default_user
from vecnadb.shared.logging_utils import get_logger
from vecnadb.modules.search.types import SearchType

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_library")
        ).resolve()
    )
    vecnadb.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_library")
        ).resolve()
    )
    vecnadb.config.system_root_directory(cognee_directory_path)

    await vecnadb.prune.prune_data()
    await vecnadb.prune.prune_system(metadata=True)

    await vecnadb.add(["TEST1"], "test1")
    await vecnadb.add(["TEST2"], "test2")

    task_1_config = {
        "vector_db_url": "cognee1.test",
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
        "vector_db_name": "",
    }
    task_2_config = {
        "vector_db_url": "cognee2.test",
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
        "vector_db_name": "",
    }

    task_1_graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": "kuzu1.db",
    }
    task_2_graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": "kuzu2.db",
    }

    # schedule both cognify calls concurrently
    task1 = asyncio.create_task(
        vecnadb.cognify(
            ["test1"], vector_db_config=task_1_config, graph_db_config=task_1_graph_config
        )
    )
    task2 = asyncio.create_task(
        vecnadb.cognify(
            ["test2"], vector_db_config=task_2_config, graph_db_config=task_2_graph_config
        )
    )

    # wait until both are done (raises first error if any)
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
