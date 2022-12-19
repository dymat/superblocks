"""Script to genereate result folder for paper
"""
import os
import sys
import geopandas as gpd

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions


path_transfer = "C:/polybox/GEC_superblocks/results"
path_result = "C:/_results_swiss_communities/_results"
path_result_paperout_blocks = os.path.join(path_transfer)
file_name = "classified_edges_withflow_0.5.shp"
hp_rw.create_folder(path_result_paperout_blocks)
hp_rw.create_folder(path_result_paperout_blocks)

cities = [
    230,
    261,
    351,
    1061,
    2701,
    3203,
    5192,
    5586,
    6621]

cities_label = {
    230: "Winterthur",
    261: "Zurich",
    351: "Bern",
    1061: "Luzern",
    2701: "Basel",
    3203: "StGallen",
    5192: "Lugano",
    5586: "Lausanne",
    6621: "Geneva"
}

write_blocks = True
write_streets = True

for city in cities:
    print("Writing out city: {}".format(city))
    city_label = cities_label[city]
    if write_blocks:
        file_path = os.path.join(path_result, str(city), "blocks", "0.5", 'block_all.shp')
        path_crs = os.path.join(path_result, str(city), "blocks", "gdf_cleaned_city_blocks.shp")

        crs_gdf = gpd.read_file(path_crs)
        gdf = gpd.read_file(file_path)

        crs = crs_gdf.crs
        gdf.crs = crs

        # Replace miniblockS by miniblock
        gdf['b_type'].loc[gdf['b_type'] == 'miniblockS'] = 'miniblock'

        gdf['b_type'].replace({"miniblock": "mini superblock"}, inplace=True)
        gdf = gdf.round()

        # Write
        gdf.to_file(os.path.join(path_result_paperout_blocks, "{}.geojson".format(city_label)), driver='GeoJSON') # GeoJSON    

print("---finished writing out all files")