"""Create shapefile with extents of all images on HD
"""

import os
import sys
import pprint
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, Point

# ZH_ RS1

path_folder = 'U:/zh'
path_out = 'U:/zh/extents'


# GeoDataFrame
crs_swisss_image = 2056

all_files = os.listdir(path_folder)
geometries = []
names = []

for file in all_files:
    # Get extend
    file_path = os.path.join(path_folder, file)
    if file_path.endswith(".tif"):
        print("File: {}".format(file))

        # Get textend
        try:
            swissimage_rs = rasterio.open(file_path)
            #pprint.pprint(swissimage_rs.meta)

            pixelSizeX = swissimage_rs.meta['transform'][0]
            pixelSizeY = -swissimage_rs.meta['transform'][4]
            #print("pixelSizeX: {}  pixelSizeY: {}".format(pixelSizeX, pixelSizeY))

            resolutionX = swissimage_rs.meta['transform'][0]
            resolutionY = -swissimage_rs.meta['transform'][4]

            x_top_left = swissimage_rs.meta['transform'][2]
            y_top_left = swissimage_rs.meta['transform'][5]
            x_bottom_right = x_top_left + (swissimage_rs.meta['width'] * resolutionX)
            y_bottom_right = y_top_left - (swissimage_rs.meta['height'] * resolutionY)

            coords = (
                (x_top_left, y_bottom_right),
                (x_bottom_right, y_bottom_right),
                (x_bottom_right, y_top_left),
                (x_top_left, y_top_left))

            geometries.append(Polygon(coords))
            names.append(file)
        except:
            print("WARNING FILE CORRUPT: {}".format(file))
            raise Exception("CORRUP")

shp_extents = gpd.GeoDataFrame(geometry=geometries, crs="epsg:{}".format(crs_swisss_image))
shp_extents['names'] = names

shp_extents.to_file(path_out)

print("-- finished createing extens for tifs --")