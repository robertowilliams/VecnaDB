from vecnadb.modules.users.models.DatasetDatabase import DatasetDatabase


def get_graph_dataset_database_handler(dataset_database: DatasetDatabase) -> dict:
    from vecnadb.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[dataset_database.graph_dataset_database_handler]
    return handler
