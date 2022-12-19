"""
Get urban green areas from Ginzler based on superblcok geometry inputs
"""
import os
import sys
import rasterio
import math
import logging
import numpy as np
import pandas as pd
import configparser
import geopandas as gpd
from progress.bar import Bar
from pyproj import Proj, transform, Transformer
from shapely.geometry import Point, MultiPoint, box
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.ops import linemerge, unary_union, polygonize
        
path_repository = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))     
sys.path.append(path_repository)

from superblocks.scripts.areal import rasterio_helper as hp_rio
from superblocks.scripts.network import helper_read_write as hp_rw

path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"
path_ch_urban_green = "C:/DATA/ginzler_greening/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95.tif"
path_GWR_out_segment = "C:/_results_swiss_communities"
temp_path = "C:/_scrap"
path_results = "C:/_results_swiss_communities/_results/"

'''path_communities = "J:/Sven/_DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"
path_ch_urban_green = "J:\Sven\_DATA\ginzler_greening/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95.tif"
path_GWR_out_segment = "J:/Sven/greening/_results_swiss_communities"
temp_path = "J:/Sven/_scrap"
path_results = "J:/Sven/greening/_results_swiss_communities/_results/"'''


hp_rw.create_folder(temp_path)

# Reference system
ginzerl_orig_crs = 2056  #epsg https://epsg.io/2056 (provided in GWR) EPSG:2056 - CH1903+ / LV95 - Projected
src_target = 21781       #epsg https://epsg.io/21781  # SwissBuildings13d: EPSG 21781

gemeinden = gpd.read_file(path_communities)
gemeinden = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren'])]
segmentation_IDs = list(gemeinden['BFS_NO'])
segmentation_IDs.sort()

case_studies = gemeinden.index.tolist()
case_studies.sort()
print("Number of gemeindens: {}".format(len(case_studies)))

case_studies = [230, 261, 351, 1061, 2701, 3203, 5192, 5586, 6621]
green_gemos = gemeinden.to_crs(ginzerl_orig_crs)

stretch_factor = "0.5"

geometry_paths = []

for segmentation_id in case_studies:
    # Add block area
    geometry_paths.append(('block', os.path.join(path_results, '{}/blocks/{}/block_all.shp'.format(segmentation_id, stretch_factor))))
    # Add street area
    geometry_paths.append(('street', os.path.join(path_results, '{}/blocks/{}/street_all.shp'.format(segmentation_id, stretch_factor))))


#output_path = "C:/_scrap/FF.tif"
            
clip_vhm_for_blocks = True

for type_def, geometry_path in geometry_paths:

    print("geometry_path: {}".format(geometry_path))

    green_gemos = gpd.read_file(geometry_path)

    # Remove invalid polygons
    green_gemos = green_gemos.loc[green_gemos.geometry.is_valid]

    green_gemos = green_gemos.to_crs(ginzerl_orig_crs)

    # Create result folder
    path_out_green = os.path.join(os.path.dirname(geometry_path), "green")
    if not os.path.exists(path_out_green):
        os.mkdir(path_out_green)

    # ===================================================
    # Read in clip geometry
    # ===================================================
    #if clip_vhm_for_blocks:
    for index_gem in green_gemos.index:
        #clip_geometry = green_gemos.loc[[index_gem]]
        #clip_geometry.to_file("C:/_scrap/A.shp")

        # ---Merge all files
        #clip_geometry_for_rasterio = unary_union(green_gemos.geometry.tolist())

        # Single polygon
        clip_geometry_for_rasterio = [green_gemos.loc[index_gem].geometry]

        # Get id
        generic_id = green_gemos.loc[index_gem]['inter_id']

        path_tif_clipped = os.path.join(path_out_green, "vhm_temp_{}_{}.tif".format(type_def, generic_id))
        path_tif_projected = os.path.join(path_out_green, "vhm_{}_{}.tif".format(type_def, generic_id))

        #print("Clip geometry crs: {}".format(clip_geometry.crs.srs))
        # ===================================================
        # Clip tif with geometry
        # ===================================================
        with rasterio.open(path_ch_urban_green) as src:
            #print("CRS of clip geometry: {}".format(green_gemos.crs.srs))
            #print("CRS tif: {}".format(src.meta['crs'].data['init']))
            assert green_gemos.crs.srs == src.meta['crs'].data['init']

            # Clip
            out_image, out_transform = rasterio.mask.mask(
                src,
                clip_geometry_for_rasterio,
                nodata=0,
                crop=True)

            out_meta = src.meta
            new_bounds = rasterio.coords.BoundingBox(
                left=out_transform[2],
                right=out_transform[2] + out_image.shape[2],
                bottom=out_transform[5] - out_image.shape[1],
                top=out_transform[5])

            # Update extent of new geotif
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

            with rasterio.open(path_tif_clipped, "w", **out_meta) as dest:
                dest.write(out_image)

            # ===================================================
            # Reproject tif to target projection https://rasterio.readthedocs.io/en/latest/topics/reproject.html
            # ===================================================
            dst_crs = 'EPSG:{}'.format(src_target)

            with rasterio.open(path_tif_clipped) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(path_tif_projected, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)

            # Rempve intermediary projection
            os.remove(path_tif_clipped)
            print("projection finished")
         
print("--finish---")
