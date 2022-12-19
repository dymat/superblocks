"""Script to genereate result folder for paper
"""
import os
import sys
import geopandas as gpd

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions


path_result = "K:/superblocks/03-results_publication/_results_superblockR1"

path_result_paperout = os.path.join(path_result, '_paper_datarepo', 'streets')
path_result_paperout_blocks = os.path.join(path_result, '_paper_datarepo', 'blocks')
file_name = "classified_edges_withflow_1.0.shp"
hp_rw.create_folder(path_result_paperout)
hp_rw.create_folder(path_result_paperout_blocks)

cities = [
    'atlanta', #
    'bankok', #
    'barcelona', #
    'berlin', #
    'budapest', #
    'cairo', #
    'hong_kong', #
    'lagos', #
    'london', #
    'madrid', #
    'melbourne', #
    'mexico_city', #
    'paris', #
    'rome', #
    'sydney', #
    'tokyo', #
    'warsaw', #
    'zurich'
    ]

write_blocks = True
write_streets = True

# Define NDI flow category values
road_flow_categories = [
    0.05,
    0.15]

for city in cities:
    print("Writing out city: {}".format(city))

    if write_blocks:
        crs_gdf = gpd.read_file(os.path.join(path_result, city, "blocks", "gdf_cleaned_city_blocks.shp"))
        gdf = gpd.read_file(os.path.join(path_result, city, "blocks", "1.0", 'block_all.shp'))

        crs = crs_gdf.crs
        gdf.crs = crs

        # Replace miniblockS by miniblock
        gdf['b_type'].loc[gdf['b_type'] == 'miniblockS'] = 'miniblock'

        #gdf['b_type'].replace({"miniblock": "mini superblock"}, inplace=True)
        #gdf['b_type'].replace({"b_miniS": "mini superblock"}, inplace=True)
        gdf['b_type'].replace({"b_miniS": "miniblock"}, inplace=True)

        # Write
        gdf.to_file(os.path.join(path_result_paperout_blocks, "{}.shp".format(city)))  #shp
        gdf.to_file(os.path.join(path_result_paperout_blocks, "{}.geojson".format(city)), driver='GeoJSON') # GeoJSON    

    if write_streets:
        df = gpd.read_file(os.path.join(path_result, city, "_flows", file_name))

        # Select only flow and classification colum to write out
        df = df[['final', 'flow_ov', 'geometry']]

        df = flow_algorithm_functions.classify_flow_cat(
            df, label_out='NDI_class', label='flow_ov', input_type='gdf', cats=road_flow_categories)

        # Rename
        df['final'] = df['final'].replace(['b_mini'], 'miniblock')
        df['final'] = df['final'].replace(['b_superb'], 'superblock')
        df['final'] = df['final'].replace(['big_road'], 'large')

        df = df.rename(columns={'final': 'class', 'flow_ov': 'NDI'})

        # Write
        df.to_file(os.path.join(path_result_paperout, "{}.shp".format(city)))  #shp
        df.to_file(os.path.join(path_result_paperout, "{}.geojson".format(city)), driver='GeoJSON') # GeoJSON    


print("---finished writing out all files")