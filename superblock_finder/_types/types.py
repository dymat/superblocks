from pydantic import BaseModel
from typing import List

class Coord(BaseModel):
    lat: float
    lng: float

class Region(BaseModel):

    class Config:
        schema_extra = {
            "example": {
                "name": "Berlin",
                "coords": [
                    Coord(lat=52.3, lng=13.5),
                    Coord(lat=53.1, lng=13.8),
                    Coord(lat=52.6, lng=13.6)
                ]
            }
        }
    coords: List[Coord]
    name: str | None

