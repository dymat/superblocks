"""
Create GWR population based on GWS dataset 2017
"""
import os
import sys
import pprint
import math
import logging
import numpy as np
import pandas as pd
import configparser
import geopandas as gpd
from progress.bar import Bar
from pyproj import Proj, transform, Transformer
from shapely.geometry import Point, MultiPoint, box

path_repository = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))      
sys.path.append(path_repository)
from vseclustering.scripts import helper as hp
from vseclustering.scripts import file_handling as basic
from vseclustering.scripts import spatial

path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"
path_gwr = "K:/STOCK-MODELLING/DATASOURCES/RAW/BuildingData/GWR/GWS2017_GEB_GWS_TMP17_20181005.txt"
path_gwr_wohnungen = "K:/STOCK-MODELLING/DATASOURCES/RAW/BuildingData/GWR/GWR_MADD_Export_MADD-20210324-A3_20210330/GWR_MADD_Export_MADD-20210324-A3_20210330/GWR_MADD_WHG-06_Data_MADD-20210324-A3_20210330.dsv"
path_BFE2021 = "C:/DATA/vse/borders/swissBOUNDARIES3D/BOUNDARIES_2021/DATEN/swissBOUNDARIES3D/SHAPEFILE_LV03_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp"
path_GWR_out_ch = "C:/_results_CH/GWR"
path_GWR_out_segment = "C:/_results_swiss_communities"

basic.create_folder(path_GWR_out_ch)

all_folders = os.listdir(path_GWR_out_segment)


# Reference systesm
gwr_proj = 2056         #epsg https://epsg.io/2056 (provided in GWR)
target_proj = 32632   #epsg https://epsg.io/21781  # SwissBuildings13d: EPSG 21781
write_anyway = False

'''bfe_2021_geometry = gpd.read_file(path_BFE2021)
bfe_2021_geometry = bfe_2021_geometry.to_crs(target_proj)
bfe_2021_geometry = bfe_2021_geometry[bfe_2021_geometry['KANTONSNUM'].notna()] 
bfe_2021_geometry = bfe_2021_geometry[bfe_2021_geometry['BFS_NUMMER'] < 9000] #remove lakes
bfe_2021_geometry = bfe_2021_geometry[bfe_2021_geometry['ICC'] == 'CH'] # Only Switzerland
segmentation_IDs = list(bfe_2021_geometry['BFS_NUMMER'])
segmentation_IDs.sort()'''

path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"
gemeinden = gpd.read_file(path_communities)
segmentation_IDs = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren'])]  #, 'Mittelzentren'])]
segmentation_IDs = list(segmentation_IDs['BFS_NO'])
segmentation_IDs.sort()
#'Grosszentren', 'Gürtel der Grosszentren', 'Gürtel der Mittelzentren', 'Mittelzentren', 'Nebenzentren der Grosszentren'
#gemeinden = gpd.read_file(path_communities)
#gemeinden = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren', 'Mittelzentren'])]
#gemeinden = gemeinden.loc[gemeinden['GDE_NO'] == 261]  #ONLY ZH
case_studies = gemeinden.index.tolist()
case_studies.sort()
print("Number of gemeindens: {}".format(len(case_studies)))

# ================================================================================
# (1) Get geometries (from Swissboundaries)
# ================================================================================
read_gwr = False
segment_and_add_gwr_flat = False
match_gwr_to_osm = True

if read_gwr:
    print("... reading in gwr buildings")

    df_gwr = pd.read_csv(path_gwr, sep=";", encoding="ISO-8859-1")

    x_values = df_gwr['GKODES'].tolist() # easting
    y_values = df_gwr['GKODNS'].tolist() # northing
    df_gwr = df_gwr.drop(columns=['DGRUNDPLZ4', 'DGRUNDPLZZ', 'GQUARTIER', 'GKODESH', 'GKODNSH'])
    building_points = []
    for x_coord, y_coord in zip(x_values, y_values):
        building_points.append(Point(x_coord, y_coord))

    # Add geometry and create gdf
    df_gwr['geometry'] = building_points
    gdf_gwr = gpd.GeoDataFrame(df_gwr, crs=gwr_proj)

    # Reproject
    gdf_gwr = gdf_gwr.to_crs(target_proj)
    print("INFO CRS: {}".format(gdf_gwr.crs))
    # ---------------------------------------
    # Matching with original BFS shape number
    # ---------------------------------------
    attributes_to_append = {'KANTONSNUM': 'int', 'BFS_NUMMER': 'int'}
    gdf = spatial.assign_polygon_to_points(
        polygons=gemeinden, #bfe_2021_geometry,
        points_or_poly=gdf_gwr,
        attributes_append=attributes_to_append)
    gdf = gdf.rename(columns={"BFS_NUMMER": "BFS_NR", "KANTONSNUM": "KT_NR"})
    gdf.to_file(os.path.join(path_GWR_out_ch, 'GWR_CH.shp'))
    print("finished reading full gwr")

if segment_and_add_gwr_flat:
    #------------------------------------
    #NOTE: GWS IS NOT THE SAME AS GWR!!!!
    #------------------------------------
    # Add Wohnungsdata to EGID building
    #read_geb_data_raw = hp.read_wohn_geb(path_gwr_wohnungen)
    # Aggregate information of all flats with same EGID
    #read_geb_data = hp.aggregate_EGID(read_geb_data_raw, columns_to_write_out=['EGID', 'WPERSHW'], vars_sum=['EGID', 'WPERSHW'])
    gdf = gpd.read_file(os.path.join(path_GWR_out_ch, 'GWR_CH.shp'))

    progress_par = Bar('Write GWR: ', max=len(segmentation_IDs))
    for segmentation_ID in segmentation_IDs:
        print("ID {}".format(segmentation_ID))
        if str(segmentation_ID) in all_folders:
            gdf_community = gdf[gdf['BFS_NR'] == int(segmentation_ID)]

            # Merge (attribute joing) of gdf
            #gwr_merged = gdf_community.merge(read_geb_data, on='EGID', how='left')
            gdf_community.to_file(os.path.join(path_GWR_out_segment, str(segmentation_ID), "GWR.shp"))
            progress_par.next()
        else:
            print("WARNING: BFE NOT FOUND: {}".format(segmentation_ID))
        progress_par.finish()

if match_gwr_to_osm:
    for folder_name in segmentation_IDs:
        print("Gemeinde: {}".format(folder_name))
        path_out = os.path.join(path_GWR_out_segment, str(folder_name), 'osm_with_gwr.shp')
        if not os.path.exists(path_out) or write_anyway:
            buildings_osm = gpd.read_file(os.path.join(path_GWR_out_segment, str(folder_name), "osm_buildings.shp"))
            buildings_gwr = gpd.read_file(os.path.join(path_GWR_out_segment, str(folder_name), "GWR.shp"))

            buildings_osm['merge_id'] = list(range(1, 1 + buildings_osm.shape[0], 1))
            osm_kd_tree = spatial.KDTree([list(a) for a in zip(buildings_osm.geometry.centroid.x.tolist(), buildings_osm.geometry.centroid.y.tolist())])
            buildings_gwr['merge_id'] = hp.gwr_to_osm(buildings_gwr, buildings_osm, osm_kd_tree, attribute='merge_id')

            # Dissolve GWR buildings with OSM building merge id (as multiple GWR per building possible)
            all_attributes = buildings_gwr.columns.tolist()
            all_attributes.remove('geometry')
            all_attributes.remove('GDENAME')
            all_attributes.remove('KT_NR')
            all_attributes.remove('BFS_NR')
            all_attributes.remove('merge_id')

            # WPERSHW : Personen mit Hauptwohnsitz
            sum_attributes = ['GAPTO']
            method_to_groupby = {}
            for attribute in sum_attributes:
                all_attributes.remove(attribute)
                if attribute in sum_attributes:
                    method_to_groupby[attribute] = 'sum' 

            # Dissolve by summing and taking the first attribute depending on row
            summed_element = buildings_gwr.groupby(by='merge_id').agg(method_to_groupby)

            gwr_dissolved = gpd.GeoDataFrame(summed_element.index.tolist(), columns=['merge_id'])

            # Add summed and dominante attributes
            for sum_attribute in sum_attributes: # summed
                gwr_dissolved[sum_attribute] = summed_element[sum_attribute].values.tolist()

            gwr_dissolved = gwr_dissolved.reset_index(drop=True)  
    
            # Merge back to osm buildings
            buildings_merged = buildings_osm.merge(gwr_dissolved, on='merge_id', how='left')
            buildings_merged.to_file(path_out)
        
    
print("--finish---")