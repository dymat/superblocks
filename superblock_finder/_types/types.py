from pydantic import BaseModel
from typing import List

class Coord(BaseModel):
    lat: float
    lng: float

class Region(BaseModel):
    coords: List[Coord]
    name: str | None


