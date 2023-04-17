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
                    Coord(lat=52.5363, lng=13.2845),
                    Coord(lat=52.5107, lng=13.2862),
                    Coord(lat=52.5168, lng=13.3726),
                    Coord(lat=52.539, lng=13.3698)
                ]
            }
        }
    coords: List[Coord]
    name: str | None

