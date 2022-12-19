"""
Calculate flow for an individual network
"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)
import geopandas as gpd

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_osm as hp_osm

# Paths
path_temp = "C:/_results_superblock/zurich"
path_network = os.path.join(path_temp, 'classified_edges.shp')
path_bb = "K:/superblocks/01-data/cities/zurich/extent.shp"
path_out = os.path.join(path_temp, '_flows')
hp_rw.create_folder(path_out)

tag_id = 'id_superb'
max_road_cap = 10
nr_help_pnts = 100

# Load bounding box
bb_shp = gpd.read_file(path_bb)
bb = hp_osm.BB(
    ymax=max(bb_shp.bounds.maxy), ymin=min(bb_shp.bounds.miny),
    xmax=max(bb_shp.bounds.maxx), xmin=min(bb_shp.bounds.minx))
bb_shp = bb.as_gdf(crs_orig=int(bb_shp.crs.srs.split(":")[1]))

# Load graph
gdf_G = gpd.read_file(path_network)
G = hp_rw.gdf_to_nx(gdf_G)

# ===============================
# Pre-steps for flow algorithm
# ===============================
# Assign flows and create
G = flow_algorithm_functions.clean_network(G)
nodes, edges = hp_rw.nx_to_gdf(G)
edges.to_file(os.path.join(path_out, "capacity_base.shp"))

# Create supergraph for flow algorithm
G, dict_G_super = flow_algorithm_functions.create_super_sinks_sources(G, bb, nr_help_pnts=nr_help_pnts)

G_base_flow = flow_algorithm_functions.flow_emund(G, dict_G_super, max_road_capacity=max_road_capacity)
nodes, edges = hp_rw.nx_to_gdf(G_base_flow)
edges.to_file(os.path.join(path_out, "base_flow_edmunds.shp"))
