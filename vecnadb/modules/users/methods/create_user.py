from uuid import UUID, uuid4
from fastapi_users.exceptions import UserAlreadyExists
from sqlalchemy.ext.asyncio import AsyncSession

from vecnadb.infrastructure.databases.relational import get_relational_engine
from vecnadb.modules.notebooks.models.Notebook import Notebook
from vecnadb.modules.notebooks.methods.create_notebook import _create_tutorial_notebook
from vecnadb.modules.users.exceptions import TenantNotFoundError
from vecnadb.modules.users.get_user_manager import get_user_manager_context
from vecnadb.modules.users.get_user_db import get_user_db_context
from vecnadb.modules.users.models.User import UserCreate
from vecnadb.modules.users.models.Tenant import Tenant

from sqlalchemy import select
from typing import Optional


async def create_user(
    email: str,
    password: str,
    is_superuser: bool = False,
    is_active: bool = True,
    is_verified: bool = False,
    auto_login: bool = False,
):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.create(
                        UserCreate(
                            email=email,
                            password=password,
                            is_superuser=is_superuser,
                            is_active=is_active,
                            is_verified=is_verified,
                        )
                    )

                    if auto_login:
                        await session.refresh(user)

                    # Update tenants and roles information for User object
                    _ = await user.awaitable_attrs.tenants
                    _ = await user.awaitable_attrs.roles

                    return user
    except UserAlreadyExists as error:
        print(f"User {email} already exists")
        raise error
