from vecnadb.modules.users.get_fastapi_users import get_fastapi_users
from vecnadb.modules.users.models.User import UserRead


def get_verify_router():
    return get_fastapi_users().get_verify_router(UserRead)
