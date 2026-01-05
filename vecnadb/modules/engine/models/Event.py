from typing import Optional, Any
from pydantic import SkipValidation
from vecnadb.infrastructure.engine import DataPoint
from vecnadb.modules.engine.models.Timestamp import Timestamp
from vecnadb.modules.engine.models.Interval import Interval


class Event(DataPoint):
    name: str
    description: Optional[str] = None
    at: Optional[Timestamp] = None
    during: Optional[Interval] = None
    location: Optional[str] = None
    attributes: SkipValidation[Any] = None

    metadata: dict = {"index_fields": ["name"]}
