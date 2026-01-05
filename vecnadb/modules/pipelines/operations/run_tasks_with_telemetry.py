import json

from vecnadb.modules.settings import get_current_settings
from vecnadb.modules.users.models import User
from vecnadb.shared.logging_utils import get_logger
from vecnadb.shared.utils import send_telemetry
from vecnadb import __version__ as cognee_version

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task


logger = get_logger("run_tasks_with_telemetry()")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, context: dict = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        async for result in run_tasks_base(tasks, data, user, context):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
        send_telemetry(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        raise error
