"""
This file provides functions to receive data for a qualitative rating of superblocks
Maintainer: Denny Mattern
"""
from abc import ABCMeta, abstractmethod

import requests
from shapely.geometry import Polygon, Point
import geopandas as gpd

from _types import Region


class Data(metaclass=ABCMeta):
    @abstractmethod
    def to_geopandas(self):
        return


class OverpassData(Data):
    def __init__(self, region: Region, overpass_url='http://overpass/api/interpreter', ):
        self.url = overpass_url
        self.data = {"elements": []}

        self._set_bbox(region)

    def _set_bbox(self, roi: Region):
        points = [(point.lat, point.lng) for point in roi.coords] + [(roi.coords[0].lat, roi.coords[0].lng)]
        self.bbox = Polygon(points).envelope.bounds

    def get_data(self):
        bbox = self.bbox
        query = f"""
            [out:json];
            // Grundschulen
            (
                way["amenity"="school"][name~"Grundschule"]{bbox}; 
                >;
                node["amenity"="school"][name~"Grundschule"]{bbox};
            ) -> .schools;
            
            
            // Kindergärten
            (
                way["amenity"="kindergarten"]{bbox}; 
                >;
                node["amenity"="kindergarten"]{bbox};
            ) -> .kindergarten;
            
            // Hausärzte
            (
                way["amenity"="doctors"](52.3, 13.5, 53.1, 13.8);
                >;
                node["amenity"="doctors"](52.3, 13.5, 53.1, 13.8);
            ) -> .amenity_doctors;
            
            ( .amenity_doctors; - 
              (
                way.amenity_doctors["healthcare"!="doctor"];
                node.amenity_doctors["healthcare"!="doctor"];
              );
            ) -> .doctors;
            
            (
                way["healthcare"="doctor"](52.3, 13.5, 53.1, 13.8);
                >;
                node["healthcare"="doctor"](52.3, 13.5, 53.1, 13.8);
                .doctors;
            ) -> .doctors;
            
            
            (
              way.doctors[!"healthcare:speciality"]; 
              node.doctors[!"healthcare:speciality"];
              way.doctors["healthcare:speciality"="general"];
              node.doctors["healthcare:speciality"="general"];
            ) -> .doctors;
            
            (.schools; .kindergarten; .doctors;) -> .result;
            (node.result[amenity]; way.result[amenity];) -> ._;
            
            out center;
        """

        r = requests.get(
            self.url,
            params={'data': query})

        print(query)
        if r.ok:
            self.data = r.json()

    def to_geopandas(self):

        for elem in self.data["elements"]:
            if "tags" not in elem.keys():
                print(elem)

        data = [[elem["tags"]["amenity"] if "amenity" in elem["tags"].keys() else "",
                 elem["tags"]["name"] if "name" in elem["tags"].keys() else "",
                 elem["tags"]["description"] if "description" in elem["tags"].keys() else "",
                 elem["lat"] if elem["type"] == "node" else elem["center"]["lat"],
                 elem["lon"] if elem["type"] == "node" else elem["center"]["lon"]
                ]
            for elem in self.data["elements"]]

        gdf = gpd.GeoDataFrame(columns=["amenity", "name", "description", "lat", "lon"], data=data)
        gdf.geometry = gdf.apply(axis=1, func=lambda row: Point(row.lon, row.lat))
        gdf.set_crs(epsg=4326, inplace=True)

        return gdf

