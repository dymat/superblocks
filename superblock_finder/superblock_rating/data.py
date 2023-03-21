"""
This file provides functions to receive data for a qualitative rating of superblocks
Maintainer: Denny Mattern
"""
from abc import ABCMeta, abstractmethod

import requests
from shapely.geometry import Polygon

from _types import Region


class Data(metaclass=ABCMeta):
    @abstractmethod
    def to_geopandas(self):
        return

    @abstractmethod
    def to_geojson(self):
        return

    @abstractmethod
    def set_bbox(self, roi: Region):
        return


class OverpassData(Data):
    def __init__(self, region:Region, overpass_url='http://overpass/api/interpreter', ):
        self.url = overpass_url
        self.data = {}

        self.set_bbox(region)

    def set_bbox(self, roi: Region):
        points = [(point.lat, point.lng) for point in roi.coords] + [(roi.coords[0].lat, roi.coords[0].lng)]
        self.bbox = Polygon(points).envelope.bounds

    @property
    def grundschulen(self):
        bbox = self.bbox #.bounds
        query = f"""
            [out:json];
            (
                way["amenity"="school"][name~"Grundschule"]{bbox}; 
                >;
                node["amenity"="school"][name~"Grundschule"]{bbox};
            ) -> .schools;
            
            .schools -> ._;
            //(way.schools[name]; node.schools[name];) -> ._;
            out center;
        """

        r = requests.get(
            self.url,
            params={'data': query})

        return r.json()

    def to_geojson(self): pass
    def to_geopandas(self): pass


