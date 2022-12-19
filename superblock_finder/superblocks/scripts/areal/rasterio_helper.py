import pprint
import json
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from affine import Affine
from pyproj import Proj, transform
from shapely.geometry import Point

from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]