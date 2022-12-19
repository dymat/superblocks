"""Assign cadastre data to communities
"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

import shutil
import geopandas as gpd

from superblocks.scripts.network import helper_osm as hp_osm

to_crs_meter = 32632 # 2056: CH1903+ / LV95   4326: WSG 84
write_anyway = True

path_cadastre_data = 'C:/DATA/amtliche_vermessung_opendata/_polygonized'
path_out = "C:/_results_swiss_communities"
path_results = os.path.join(path_out, "_results")

path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"
gemeinden = gpd.read_file(path_communities)
gemeinden = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren'])]  #, 'Mittelzentren'])]
segmentation_IDs = list(gemeinden['BFS_NO'])
segmentation_IDs.sort()
segmentation_IDs = [1061, 3203, 5192, 5586]

for segmentation_ID in segmentation_IDs:
    print("segmentation_ID: {}".format(segmentation_ID))
    # Get cadastredata
    source = os.path.join(path_cadastre_data, "{}.XXX".format(segmentation_ID))

    # Copy to folder
    path_bfe_folder = os.path.join(path_out, str(segmentation_ID))
    destination = os.path.join(path_bfe_folder, 'cadastre.shp')
    files_to_copy = [
        os.path.join(path_bfe_folder, 'cadastre.shp'),
        os.path.join(path_bfe_folder, 'cadastre.cpg'),
        os.path.join(path_bfe_folder, 'cadastre.dbf'),
        os.path.join(path_bfe_folder, 'cadastre.prj'),
        os.path.join(path_bfe_folder, 'cadastre.shx')]

    out_path = os.path.join(path_bfe_folder, "gdf_cadastre_no_buildings.shp")
    if os.path.exists(source.replace('XXX', 'shp')):
        if not os.path.exists(out_path) or write_anyway:
            for file_to_copy in files_to_copy:
                dest = shutil.copyfile(source.replace('XXX', file_to_copy[-3:]), file_to_copy)

            # Prepare cadastre
            buildings = gpd.read_file(os.path.join(path_bfe_folder, "osm_buildings.shp"))
            gdf_cadastre = gpd.read_file(destination)
            gdf_cadastre = gdf_cadastre.to_crs(to_crs_meter)

            # --Get very large cadasre plots which are something else
            large_area_crit = 100000  # [m2]
            gdf_cadastre_very_large = gdf_cadastre.loc[gdf_cadastre.area > large_area_crit]
            gdf_cadastre_very_large.to_file(os.path.join(path_bfe_folder, "gdf_cadastre_very_large.shp"))

            # --Select only streets
            buildings['geometry'] = buildings.geometry.buffer(-2)
            buildings = buildings.loc[~buildings.is_empty]
            buildings = buildings.loc[buildings.geometry.is_valid]
            buildings = buildings.loc[buildings.geometry.type == 'Polygon']
            gdf_cadastre_no_buildings = hp_osm.remove_buildng_plots(gdf_cadastre, buildings)
            gdf_cadastre_no_buildings.to_file(out_path)
    else:
        print("CADASTRE NOT AVAILABE: {}".format(source))
