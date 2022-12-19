"""Load actual traffic data and transfer to OSM data

NVPM is the attribute to look for
"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

import geopandas as gpd

from superblocks.scripts.network import helper_network as hp_net
from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import helper_osm as hp_osm

to_crs_meter = 32632

# Paths
path_out = "C:/_results_superblock/zurich/_flows"
path_traffic_flow_2056 = "K:/superblocks/01-data/traffic_data/traffic_flow_raw.shp"
path_traffic_flow_32632 = "K:/superblocks/01-data/traffic_data/traffic_flow_32632.shp"
path_osm_street_files = 'C:/_results_swiss_communities'

# Path to base flow
path_own_network = "C:/_results_superblock/zurich/_flows/base_flow_edmunds_recalculated.shp"

project_streets = False
assign_flow_attributes = True

if project_streets:
    street_raw = gpd.read_file(path_traffic_flow_2056)
    street_raw = hp_net.project(street_raw, to_crs_meter)
    street_raw.to_file(path_traffic_flow_32632)

if assign_flow_attributes:

    # Load flow graph
    traffic_data = gpd.read_file(path_traffic_flow_32632)
    print("Traffic cata crs: {}".format(traffic_data.crs.srs))

    all_files = os.listdir(path_osm_street_files)

    for all_file in all_files:
        path_osm_street = os.path.join(path_osm_street_files, all_file, 'street_network_edges_with_attributes_pop_density.shp')

        # Assign traffic flow to own network
        gdf_street = gpd.read_file(path_osm_street)
        print("transfer crs: {}".format(gdf_street.crs.srs))
        G_street = hp_rw.gdf_to_nx(gdf_street)
        bb = hp_osm.BB(
            ymax=max(gdf_street.geometry.bounds.maxy),
            ymin=min(gdf_street.geometry.bounds.miny),
            xmax=max(gdf_street.geometry.bounds.maxx),
            xmin=min(gdf_street.geometry.bounds.minx))
        bb_gdf = bb.as_gdf(crs_orig=to_crs_meter)

        traffic_data_clip = hp_net.clip_outer_polygons(traffic_data, bb_gdf.geometry[0])

        G_base = hp_rw.gdf_to_nx(traffic_data_clip)

        # Clip network
        # DWV_FZG:  Verkehrsbelastung, alle Fahrzeuge, durchschnittlicher Werktagverkehr (DWV)
        # DWV_PW:   Verkehrsbelastung, Personenwagen, durchschnittlicher Werktagverkehr (DWV)
        labels = ['DWV_FZG', 'DWV_PW']
        G_assigned = hp_net.assign_attribute_by_largest_intersection(
            G_street,
            G_base,
            min_intersection_d=4,
            crit_buffer=4,
            labels=labels)
        _, edges = hp_rw.nx_to_gdf(G_assigned)

        edges.to_file(os.path.join(path_osm_street_files, all_file, "G_with_traffic_raw.shp"))


print("____finish___")
    