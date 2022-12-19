"""Helper functions
"""
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



def rasterio_reproject_function(path_tif_from, path_reprojected, src_target):
    """
    https://rasterio.readthedocs.io/en/latest/topics/reproject.html             
    """
    print("... projecting")
    with rasterio.open(path_tif_from) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            src_target,
            src.width,
            src.height,
            *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'epsg:{}'.format(src_target),
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(path_reprojected, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src_target,
                    resampling=Resampling.nearest)

    print("projection finished: {}".format(path_reprojected))
