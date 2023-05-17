"""
This file provides functions to receive data for a qualitative rating of superblocks
Maintainer: Denny Mattern
"""
from abc import ABCMeta, abstractmethod

import geopandas as gpd
import requests
from _types import Region
from shapely.geometry import Polygon, Point


class Data(metaclass=ABCMeta):
    @abstractmethod
    def to_geopandas(self):
        return

    @abstractmethod
    def evaluate(self, blocks: gpd.GeoDataFrame):
        """
        Evaluates this data against a list of superblocks
        :param blocks: list of superblocks
        :return: gpd.GeoDataFrame
        """
        return gpd.GeoDataFrame


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
            ///////////////////////////
            // BILDUNG
            ///////////////////////////
            // Grundschulen
            (
                way["amenity"="school"][name~"Grundschule"]{bbox}; 
                >;
                node["amenity"="school"][name~"Grundschule"]{bbox};
            ) -> .schools;
            
            (node.schools[amenity]; way.schools[amenity];) -> .result;
                        
            
            // Kindergärten
            (
                way["amenity"="kindergarten"]{bbox}; 
                >;
                node["amenity"="kindergarten"]{bbox};
            ) -> .kindergarten;
            
            (.result; node.kindergarten[amenity]; way.kindergarten[amenity];) -> .result;
            
            ///////////////////////////
            // MEDIZINISCHE VERSORGUNG
            /////////////////////////// 
                       
            // Ärzte
            (
                way["amenity"="doctors"]{bbox};
                >;
                node["amenity"="doctors"]{bbox};
            ) -> .amenity_doctors;
            
            ( .amenity_doctors; - 
              (
                way.amenity_doctors["healthcare"!="doctor"];
                node.amenity_doctors["healthcare"!="doctor"];
              );
            ) -> .doctors;
            
            (
                way["healthcare"="doctor"]{bbox};
                >;
                node["healthcare"="doctor"]{bbox};
                .doctors;
            ) -> .doctors;
            
            (.result; node.doctors[amenity]; way.doctors[amenity];) -> .result;
            
            // Apotheken
            (
                way["amenity"="pharmacy"]{bbox};
                >;
                way["healthcare"="pharmacy"]{bbox};
                >;
                node["amenity"="pharmacy"]{bbox};
                node["healthcare"="pharmacy"]{bbox};
            ) -> .pharmacy;
            
            (.result; node.pharmacy[amenity]; way.pharmacy[amenity];) -> .result;

            ///////////////////////////
            // ESSEN / TRINKEN
            ///////////////////////////
            
            (
                way["amenity"="bar"]{bbox};
                >;
                node["amenity"="bar"]{bbox};
                
                way["amenity"="cafe"]{bbox};
                >;
                node["amenity"="cafe"]{bbox};
                
                way["amenity"="pub"]{bbox};
                >;
                node["amenity"="pub"]{bbox};
                
                way["amenity"="biergarten"]{bbox};
                >;
                node["amenity"="biergarten"]{bbox};
                
                way["amenity"="ice_cream"]{bbox};
                >;
                node["amenity"="ice_cream"]{bbox};
                
                way["amenity"="restaurant"]{bbox};
                >;
                node["amenity"="restaurant"]{bbox};
                
            ) -> .cafe_bar_restaurant;
            
            (.result; node.cafe_bar_restaurant[amenity]; way.cafe_bar_restaurant[amenity];) -> .result;
            
            (                
                way["shop"="bakery"]{bbox};
                >;
                node["shop"="bakery"]{bbox};
            ) -> .bakery;
            
            (.result; node.bakery[shop]; way.bakery[shop];) -> .result;

            
            .result -> ._;
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

    def __len__(self):
        return len(self.data["elements"])

    def evaluate(self, blocks: gpd.GeoDataFrame):
        gdf = self.to_geopandas()

