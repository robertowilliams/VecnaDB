from pydantic import Field
from vecnadb.infrastructure.engine import DataPoint
from vecnadb.modules.engine.models.Timestamp import Timestamp


class Interval(DataPoint):
    time_from: Timestamp = Field(...)
    time_to: Timestamp = Field(...)
