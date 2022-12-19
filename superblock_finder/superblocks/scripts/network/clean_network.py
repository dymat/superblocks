"""
Cleaning networks
Note: Edege node is always - 1
https://www.timlrx.com/2019/01/05/cleaning-openstreetmap-intersections-in-python/

Ideas
[ ] Intersections removing
[ ] Merge very cloe spoints (but only end-points of graphs, no intermediate points --> simple graph)
[ ] Self-loops
[ ] Remove parallel lines
[ ] Centerlines https://github.com/fitodic/centerline

https://github.com/tomalrussell/snkit/blob/master/notebooks/snkit-demo.ipynb

https://www.timlrx.com/2019/01/05/cleaning-openstreetmap-intersections-in-python/


"""
import os
import sys
print(sys.executable)
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd

to_crs = 2056  # 2056: CH1903+ / LV95   4326: WSG 84

# Add path to main module
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)


# Imports
from superblocks.scripts.network import helper_network as hp_net

# Paths
#path_roads = "K:/superblocks/01-data\example_data/osm_roads_4326.shp"   # path to road network
path_roads = "K:/superblocks/01-data/cities/zurich/street_network_edges.shp"   # path to road network
path_temp = "K:/superblocks/01-data/_scrap"       # temp path


# -----------------------
# Load shapfile and clean
# -----------------------

# MultieLineString to Linestring and remove intersections TODO: IMPROVE DO NOT CONVERT WHOLE BUt ONLY INTERSECTING NODES
gdf_road = gpd.read_file(path_roads)
gdf_road = hp_net.gdf_multilinestring_to_linestring(gdf_road)
gdf_road = hp_net.clean_up_intersections(gdf_road)
gdf_road.to_file(os.path.join(path_temp, "cleandnetw.shp"))


#G = gpd.read_file(path_roads)
G = nx.read_shp(os.path.join(path_temp, "cleandnetw.shp"), strict=True)
G.graph['crs'] = "epsg:{}".format(to_crs)  # Add CRS to graph
 
# Remove self loops
G.remove_edges_from(nx.selfloop_edges(G))

# Remove all nodes with no edges (isolates)
solate_nodes = list(nx.isolates(G))
G.remove_nodes_from(solate_nodes)
print("Number of edges: {}".format(G.number_of_edges()))
print("Number of nodes: {}".format(G.number_of_nodes()))

# TEMP
#nodes, edges, sw = momepy.nx_to_gdf(G, points=True, lines=True, spatial_weights=True)

#nodes.to_file(os.path.join(path_temp, '_street_temp_nodes.shp'))
#edges.to_file(os.path.join(path_temp, '_street_temp_edges.shp'))
#G = nx.read_shp(os.path.join(path_temp, '_street_temp_edges.shp'), strict=True)
#G.graph['crs'] = "epsg:{}".format(to_crs)  # Add CRS to graph

# Contract nodes which are close by
# = nx.contracted_nodes(G, list(G.nodes)[541], list(G.nodes)[539], self_loops=False)

# ------Merge close nodes
#G = hp_net.clean_merge_close(G, list(G.nodes)[541], list(G.nodes)[539])



#G = hp_net.clean_merge_close(G, merge_distance=10)
print("Final Number of edges: {}".format(G.number_of_edges()))
print("Final Number of nodes: {}".format(G.number_of_nodes()))


# ----------------------------
# plot the cleaned-up intersections
# ------------------------------
#nodes, edges, sw = momepy.nx_to_gdf(G, points=True, lines=True, spatial_weights=True)

nodes, edges = hp_rw.nx_to_gdf(G)
        
fig, ax = plt.subplots()
nodes.plot(ax=ax)
edges.plot(ax=ax)

for x, y, label in zip(nodes.geometry.x, nodes.geometry.y, nodes.nodeID):
    ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    
plt.show()



print("---finished____")