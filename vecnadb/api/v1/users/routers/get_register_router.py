from vecnadb.modules.users.get_fastapi_users import get_fastapi_users
from vecnadb.modules.users.models.User import UserRead, UserCreate


def get_register_router():
    return get_fastapi_users().get_register_router(UserRead, UserCreate)
