"""
"""
import os
import sys

from networkx.algorithms.shortest_paths.unweighted import predecessor

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)
import math
import logging
import json
import math
import pandas as pd
import numpy as np
import math
import itertools
import copy
import pprint
import shapely
import shapely.wkt
import networkx as nx
from rtree import index
from progress.bar import Bar
import geopandas as gpd
from haversine import haversine, Unit
from shapely.geometry import MultiLineString, LineString, Point, point, shape, Polygon, mapping, box
from shapely.ops import transform, linemerge, linemerge, unary_union, polygonize
import pyproj
import itertools
from networkx.algorithms.connectivity import local_edge_connectivity
from shapely.geometry import Polygon, Point

from superblocks.scripts.network import helper_read_write as hp_rw


def read_ids_and_network(nested_dict, G_in):
    """Get all the different individual interventions per ID
    """
    containder_id_graphs = {}

    all_ids = []
    for edge, keys in nested_dict.items():
        for key in keys:
            all_ids.append(key)
    all_ids = list(set(all_ids))

    for id in all_ids:
        G = nx.Graph()
        G.graph['crs'] = G_in.graph['crs']
        containder_id_graphs[id] = G

    for id in all_ids:
        G_single = containder_id_graphs[id]
        for edge, keys in nested_dict.items():
            for key in keys:
                if key == id:
                    G_single.add_edge(edge[0], edge[1])
                    edge_attributes = G_in.edges[edge[0], edge[1]]
                    for key, value in edge_attributes.items():
                        G_single.edges[(edge[0], edge[1])][key] = value

        containder_id_graphs[id] = G_single

    return containder_id_graphs
        







def flatten_geometry(input_gdf):
    """
    Remove LineStringZ geometry objects and flatten
    """
    for input_index in input_gdf.index:
        geom_with_z = input_gdf.loc[input_index].geometry
        geom_flat = LineString([xy[0:2] for xy in list(geom_with_z.coords)]) 
        input_gdf.loc[input_index, 'geometry'] = geom_flat

    return input_gdf


def nr_edges_34_subgraph(G_intervention, degree=3):
    """Count number of public spaces (edges with number of neighbours >= 3)
    """
    nr_nodes_neig34 = 0

    for node in G_intervention:
        nr_neigbors = list(G_intervention.neighbors(node))
        if len(nr_neigbors) >= degree:
            nr_nodes_neig34 += 1

    return nr_nodes_neig34


def nr_edges_34_cycle(G, cycle, degree=3):
    """Count number of public spaces (edges with number of neighbours >= 3)
    """
    nr_nodes_neig34 = 0

    for node in cycle:
        nr_neigbors = list(G.neighbors(node))
        if len(nr_neigbors) >= degree:
            nr_nodes_neig34 += 1

    return nr_nodes_neig34


def dock_to_path(edge_full_details, additional_path):
    """Add two path that they are joined
    """
    if len(edge_full_details) > 0:
        prev_node = edge_full_details[-1]
        from1 = additional_path[0]
        from2 = additional_path[-1]
        dist_prev_1 = math.hypot((prev_node[0] - from1[0])**2 + (prev_node[1] - from1[1])**2)
        dist_prev_2 = math.hypot((prev_node[0] - from2[0])**2 + (prev_node[1] - from2[1])**2)

        # invert first line if not close to any end, swap
        if dist_prev_1 != 0 and dist_prev_2 != 0:
            edge_full_details = edge_full_details[::-1]
            prev_node = edge_full_details[-1]
            dist_prev_1 = math.hypot((prev_node[0] - from1[0])**2 + (prev_node[1] - from1[1])**2)
            dist_prev_2 = math.hypot((prev_node[0] - from2[0])**2 + (prev_node[1] - from2[1])**2)
        
        if dist_prev_1 > dist_prev_2:
            additional_path = additional_path[::-1] #invert

        additional_path = additional_path[1:] # Remove first element
    edge_full_details.extend(additional_path)

    return edge_full_details


def generate_fishnet(regional_bb, crit_bb_width=1000):
    """Based on global bounding box,
    split the global bounding box into several
    local bounding boxes
    https://gist.github.com/lossyrob/7b620e6d2193cb55fbd0bffacf27f7f2
    """
    local_bbs = []
    geometry = regional_bb.geometry[0]

    # Global bounding box
    bounds  = geometry.bounds
    xmin = bounds[0]
    xmax = bounds[2]
    ymin = bounds[1]
    ymax = bounds[3]

    # Width
    width_x = (xmax - xmin)
    width_y = (ymax - ymin)

    # Is optimized depending on 
    cols = round(width_x / crit_bb_width) #math.ceil(width_x / crit_bb_width)
    rows = round(width_y / crit_bb_width) #math.ceil(width_y / crit_bb_width)

    if cols == 0:
        cols = 1
    if rows == 0:
        rows = 1

    print("Resulting width: {} hight: {}".format(width_x / cols, width_y / cols))

    # Create fishnet
    dx = (xmax - xmin) / cols
    dy = (ymax - ymin) / rows
    cut_geoms = {}
    for row in range(0, rows):
        for col in range(0, cols):
            b = box(xmin + (col * dx),
                    ymax - ((row + 1) * dy),
                    xmin + ((col + 1) * dx),
                    ymax - (row * dy))
            g = geometry.intersection(b)
            if not g.is_empty:
                cut_geoms[(col, row)] = g

    base_name = 'local_grid'
    for (col, row), geom in cut_geoms.items():
        gdf_bb_reg = gpd.GeoDataFrame([geom], columns=['geometry'], crs=regional_bb.crs)
        local_bbs.append(gdf_bb_reg)

    return local_bbs


def create_final_classification(
        G,
        pedestrianstreets,
        big_road_labels,
        crit_consider_bus
    ):
    """Final classification (superblock first, then miniblock)
    """
    # Classify big road first
    nx.set_edge_attributes(G, None, "big_road")
    for edge in G.edges:
        tag_highway = G.edges[edge]['tags.highway']
        crit_tram = G.edges[edge]['tram']
        crit_trolleybus = G.edges[edge]['trolleybus']
        crit_bus = G.edges[edge]['bus']
        if not crit_consider_bus:
            crit_bus = False
        if tag_highway in big_road_labels or crit_bus or crit_tram or crit_trolleybus:
            G.edges[edge]['big_road'] = 1

    # Classify
    nx.set_edge_attributes(G, None, "final")
    for edge in G.edges:
        tag_highway = G.edges[edge]['tags.highway']
        b_superb = G.edges[edge]['b_superb']
        b_mini = G.edges[edge]['b_mini']
        b_miniS = G.edges[edge]['b_miniS']
        big_road = G.edges[edge]['big_road']
        highway_type = G.edges[edge]['tags.highway']

        # Classification decision tree
        if big_road == 1:
            final_class = 'big_road'
        elif highway_type == 'service':
            final_class = 'service'

        else:
            if b_superb == 1:
                final_class = 'b_superb'
            else:
                if b_mini == 1:
                    final_class = 'b_mini'
                else:
                    if b_miniS == 1:
                        final_class = 'b_miniS'
                    else:
                        if highway_type in pedestrianstreets:
                            final_class = 'pedestrian'
                        else:
                            final_class = 'other'

        # Rewrite other classification (4.10.2021) (Rewrite all others)
        if final_class == 'b_superb':
            G.edges[edge]['b_mini'] = None
            G.edges[edge]['b_miniS'] = None
            G.edges[edge]['id_mini'] = None
            G.edges[edge]['id_miniS'] = None
        if final_class == 'b_mini':
            G.edges[edge]['b_miniS'] = None
            G.edges[edge]['id_miniS'] = None

        # Classification
        G.edges[edge]['final'] = final_class

    return G


def create_subgraph_interior_roads_superblock(cycle, nodes_neighbouring, G):
    """Create subgraph of all inner roads but exclude edges which connect neighbours only
    """
    all_nodes = cycle + nodes_neighbouring
    subgraph_inner_edges = G.subgraph(all_nodes).copy()
    for edge in subgraph_inner_edges.edges:  # if not an edge going through the node, remove edge
        if (edge[0] not in cycle and edge[1] not in cycle):
            subgraph_inner_edges.remove_edge(edge[0], edge[1])

    return subgraph_inner_edges


def create_subgraph_interior_roads_miniblock(node_4, nodes_neighbouring, G):
    """Create subgraph of all inner roads but exclude edges which connect neighbours only
    """
    nodes_neighbouring.append(node_4)
    subgraph_inner_edges = G.subgraph(nodes_neighbouring).copy()
    for edge in subgraph_inner_edges.edges:   # if not an edge going through the node, remove edge
        if (edge[0] != node_4 and edge[1] != node_4):
            subgraph_inner_edges.remove_edge(edge[0], edge[1])

    return subgraph_inner_edges


def get_subgraph_degree(
        G,
        degree_nr,
        method='node_removal',
        crit_exact=False
    ):
    """Create a usgraph on, ly having a network
    consisting of nodes with a defined degree
    """
    G_deg = G.copy()

    if method == 'node_removal':
        for node in G.nodes():
            #if node == (464699.4628167151, 5245531.730793313):
            #    print("--")
            degree = G.degree[node]
            if crit_exact:
                if degree == degree_nr:
                    pass
                else:
                    G_deg.remove_node(node)
            else:
                if degree >= degree_nr:
                    pass
                else:
                    G_deg.remove_node(node)

        list_exact_degree_nodes = list(G_deg.nodes)

    if method == 'edge_neighbours':
        edges_to_keep = []
        list_exact_degree_nodes = []
        for node in G.nodes():
            degree = G.degree(node)
            if degree >= degree_nr:
                neighbours = list(G.neighbors(node))  # Same as list(G.edges(node))
                list_exact_degree_nodes.append(node)
                assert len(neighbours) >= degree_nr
                for neighbor in neighbours:
                    edges_to_keep.append((node, neighbor))

        for edge in G.edges:
            if edge in edges_to_keep or (edge[1], edge[0]) in edges_to_keep:
                pass
            else:
                if edge in G_deg.edges:
                    G_deg.remove_edge(edge[0], edge[1])

    # Remove isolated nodes
    G_deg.remove_nodes_from(list(nx.isolates(G_deg)))

    return G_deg, list(list_exact_degree_nodes)


def remove_nodes_degree(G_in, degree):
    """Remove nodes with certain degree
    """
    G = G_in.copy()

    nodes_to_drop = []
    for node in G.nodes:
        if G.degree(node) == degree:
            nodes_to_drop.append(node)
    for node in nodes_to_drop:
        G.remove_node(node)

    return G


def get_superblock_cycles(G_4x4_orig, nodes_larger4degree, crit_circle_max_length):
    """Get all 4x4 nodes and create subgraph only with 4x4 edges
    Find cycles
    """
    G_4x4 = G_4x4_orig.copy()
    for edge in G_4x4.edges:
        G_4x4.edges[edge]['m_distance'] = G_4x4.edges[edge]['geometry'].length

    # Number of shortes paths which are checked
    nr_of_shortest_paths = 2
    cycles_4x4 = []

    bar = Bar("Get superblock cycles", max=len(nodes_larger4degree))
    for node_4_4 in nodes_larger4degree:
        nodes_neighbouring = list(G_4x4.neighbors(node_4_4))
        #if node_4_4 == (932970.674625253537670, 3957308.723676646593958):
        #    print("---")
        # Test if a within neibhouring distance, a circle of node_4_4
        for neigbor in nodes_neighbouring:
            temp_G = nx.Graph() # Temp graph to store delted edges and nodes

            # Test if shortest path only on node_4_4
            edge_attributes = G_4x4.edges[(node_4_4, neigbor)]
            length_fist_edge_not_in_path = edge_attributes['geometry'].length

            temp_G.add_edge(node_4_4, neigbor)
            edge_attributes = G_4x4.edges[edge]
            for key, value in edge_attributes.items():
                temp_G.edges[(node_4_4, neigbor)][key] = value
            G_4x4.remove_edge(node_4_4, neigbor)

            paths_checked = 0
            while nr_of_shortest_paths > paths_checked:
                path_found = False
                try:
                    # Select first shortest paths (only several if identical length)
                    shortest_path = list(nx.all_shortest_paths(G_4x4, source=neigbor, target=node_4_4, weight='m_distance'))[0]
                    path_found = True
                except Exception as e: # Now path found
                    paths_checked = nr_of_shortest_paths

                if path_found:
                    paths_checked += 1
                    distance_path_neigbor_to_source = get_distance_along_path(G_4x4, shortest_path)
                    distance_full_path = distance_path_neigbor_to_source + length_fist_edge_not_in_path

                    # Remove shortest path
                    shortest_path_edges = to_tuple_list(shortest_path)
                    for edge in shortest_path_edges:

                        temp_G.add_edge(edge[0], edge[1])
                        edge_attributes = G_4x4.edges[edge]
                        for key, value in edge_attributes.items():
                            temp_G.edges[edge][key] = value
                        G_4x4.remove_edge(edge[0], edge[1])

                    if distance_full_path > crit_circle_max_length: # circle too long
                        continue
                    else:
                        shortest_path.insert(0, node_4_4)
                        cycles_4x4.append(shortest_path)

            # Compose graph again
            G_4x4 = nx.compose(G_4x4, temp_G)

        bar.next()
    bar.finish()

    # ----------------------
    # ---Remove duplicate cycles
    # ----------------------
    print("Number of all cycles: {}".format(len(cycles_4x4)))
    unique_cycles = []
    list_centroids = []
    while len(cycles_4x4) > 0:
        cycle_to_test = cycles_4x4.pop()
        centroid_geom = Polygon(cycle_to_test).centroid
        centroid = (centroid_geom.x, centroid_geom.y)
        if centroid not in list_centroids:
            unique_cycles.append(cycle_to_test)
            list_centroids.append(centroid)

    print("Number of unique cycles: {}".format(len(unique_cycles)))
    return unique_cycles



def clip_streets(superblock, streets):
    """ Clip street area from entire block
    """
    assert superblock.shape[0] == 1
    superblock_geom = superblock.geometry[0]
    
    # Substract mutliple street areas
    for geom_street in streets.geometry:
        superblock_no_streets = superblock_geom.difference(geom_street)


        superblock_geom = superblock_no_streets

    if superblock_no_streets.type == 'MultiPolygon':
        gdf_superblock_no_streets = gpd.GeoDataFrame(superblock_no_streets, columns=['geometry'], crs=streets.crs)
    elif superblock_no_streets.type == 'Polygon':
        gdf_superblock_no_streets = gpd.GeoDataFrame([superblock_no_streets], columns=['geometry'], crs=streets.crs)
    elif superblock_no_streets.type == 'GeometryCollection':
        superblock_no_streets_only_polygons = []
        print(len(superblock_no_streets))
        for i in list(superblock_no_streets):
            if i.type == 'Polygon':
                superblock_no_streets_only_polygons.append(i)
        gdf_superblock_no_streets = gpd.GeoDataFrame([superblock_no_streets_only_polygons], columns=['geometry'], crs=streets.crs)
    else:
        raise Exception("Invalid type: {}".format(superblock_no_streets.type))
    return gdf_superblock_no_streets

'''def clip_negative(superblock, superblock_negative):
    """
    """
    superblock_geom = superblock.geometry[0]
    superblock_negative_geom = 

    
    # Substract mutliple street areas
    cleaned_streets = []
    for geom_street in streets.geometry:
        cleaned_street = geom_street.difference(superblock_geom)
        cleaned_streets.append(cleaned_street)

    if cleaned_streets.type == 'MultiPolygon':
        gdf_cleaned_streets = gpd.GeoDataFrame(cleaned_streets, columns=['geometry'], crs=streets.crs)
    elif cleaned_streets.type == 'Polygon':
        gdf_cleaned_streets = gpd.GeoDataFrame([cleaned_streets], columns=['geometry'], crs=streets.crs)
    else:
        raise Exception("Invalid type")
        
    #if polygon.intersects(clip_polygon):
    #       polygon = polygon.difference(clip_polygon)
            
    return gdf_cleaned_streets
'''
def create_city_blocks(gdf_street):
    """
    """
    merged = linemerge(gdf_street.geometry.to_list())
    borders = unary_union(merged)
    city_blocks = polygonize(borders)
    gdf_city_blocks = gpd.GeoDataFrame(city_blocks, columns=['geometry'], crs=gdf_street.crs)
    gdf_city_blocks = gdf_city_blocks.reset_index(drop=True)

    # Create urban structure units based on street network
    '''city_blocks = polygonize(gdf_street.geometry.to_list())
    gdf_city_blocks = gpd.GeoDataFrame(city_blocks, columns=['geometry'], crs=gdf_street.crs)
    gdf_city_blocks = gdf_city_blocks.reset_index(drop=True)'''

    return gdf_city_blocks


def clean_city_blocks(city_blocks, gdf_other_plots):
    """Clip away polygons from generated urban structure units
    """
    rTree = index.Index()
    for index_block in city_blocks.index:
        geometry = city_blocks.loc[index_block].geometry
        rTree.insert(index_block, geometry.bounds)

    id_to_remove = []
    for index_other in gdf_other_plots.index:
        geom_to_remove = gdf_other_plots.loc[index_other].geometry

        # Get intersecting cityblocks
        intersection_blocks = list(rTree.intersection(geom_to_remove.bounds))

        for intersection_block in intersection_blocks:
            geom_block = city_blocks.iloc[intersection_block].geometry

            # Clip away
            try:
                if geom_block.intersects(geom_to_remove):
                    '''geom_block_clipped = geom_block.difference(geom_to_remove)
                    dist_clean_buffer = 2
                    geom_block_clipped = geom_block_clipped.buffer(dist_clean_buffer * -1)
                    geom_block_clipped = geom_block_clipped.buffer(dist_clean_buffer)
                    if geom_to_remove.contains(geom_block):
                        id_to_remove.append(intersection_block)
                    elif geom_block_clipped.area > 0:
                        city_blocks.at[intersection_block, 'geometry'] = geom_block_clipped
                    else:
                        pass'''

                    # Remove entirely
                    if geom_to_remove.contains(geom_block):
                        id_to_remove.append(intersection_block)
                    elif geom_to_remove.within(geom_block):
                        id_to_remove.append(intersection_block)
            except:
                print("... coulcdn't clip because of tpological error?")

    city_blocks = city_blocks.drop(index=id_to_remove)
    city_blocks = city_blocks.reset_index(drop=True)

    return city_blocks


'''
def clean_interventions(intervention_gdf):
    """If intervention is multipolygon, then only
    select the largest intervention

    Needs cleaning, because some part of a superblock
    may already be pedestrian or living streets and
    therefore multipolygon
    """
    intervention_gdf = intervention_gdf[['geometry']]
    buffer_size = 1  # m
    gdf_intervention_buffer = intervention_gdf.buffer(buffer_size)
    
    gdf_intervention_union = gpd.GeoDataFrame(
        [unary_union(gdf_intervention_buffer.geometry.tolist())], columns=['geometry'], crs=gdf_intervention_buffer.crs)

    intervention_gdf = gpd.GeoDataFrame(
        #unary_union(intervention_gdf.geometry.tolist()),
        MultiLineString(intervention_gdf.geometry.tolist()),
        columns=['geometry'],
        crs=intervention_gdf.crs)

    if intervention_gdf.shape[0] > 1:
        all_length = list(intervention_gdf.geometry.length)
        index_longest_lenght = all_length.index(max(all_length))
        intervention_gdf = intervention_gdf.iloc[[index_longest_lenght]]
    else:
        pass

    return intervention_gdf
'''
def get_intersecting_buildings(buildings, block):
    """Select buildings which are in block
    """
    index_list = []
    block_geom = block.geometry[0]
    for index_build in buildings.index:
        building_geom = buildings.loc[index_build].geometry

        if block_geom.contains(building_geom):
            index_list.append(index_build)
    selection_buildings = buildings.loc[index_list]

    return selection_buildings


def spatial_select_fullblock(
        gdf_intervention,
        gdf_city_blocks,
        min_intersection_a=10
    ):
    """Create superbock based on urban structure units.
    Buffer the streets with small buffer and then select all USU
    which intersect based on the min_intersection_a
    """
    # Buffer streets and merge them
    buffer_width = 1  # m
    gdf_intervention_buffer = gdf_intervention.buffer(buffer_width)
    union_geom = unary_union(gdf_intervention_buffer.geometry.tolist())

    if union_geom.geom_type == 'Polygon':
        gdf_intervention_union_geom = union_geom
    elif union_geom.geom_type == 'MultiPolygon':
        all_areas = list(i.area for i in union_geom)
        index_max_area = all_areas.index(max(all_areas))
        gdf_intervention_union_geom = list(union_geom)[index_max_area]    
    else:
        raise Exception("More than ony geometry")

    #gdf_intervention_union = gpd.GeoDataFrame(union_geom, columns=['geometry'], crs=gdf_intervention.crs)
    #gdf_intervention_union_geom = gdf_intervention_union.geometry[0]

    # Create search tree for city blocks Note: could be speed up by taking it out
    gdf_city_blocks = gdf_city_blocks.reset_index(drop=True)

    # Iterate city blocks and check whether enough overlap to count for superblock
    city_blocks_to_merge = []
    for city_block_id in gdf_city_blocks.index:
        city_block_geom = gdf_city_blocks.loc[city_block_id].geometry
        if city_block_geom.intersects(gdf_intervention_union_geom):
            intersection_area = city_block_geom.intersection(gdf_intervention_union_geom).area

            if intersection_area > min_intersection_a:
                city_blocks_to_merge.append(city_block_id)

    superblock = gdf_city_blocks.loc[city_blocks_to_merge]

    # ---Merge cityblock
    if superblock.shape[0] > 0:
        merged_cityblock = unary_union(superblock.geometry.tolist())
        if merged_cityblock.type == 'MultiPolygon':
            merged_cityblock = hp_rw.get_largest_polygon(merged_cityblock)
        superblock_merged = gpd.GeoDataFrame([merged_cityblock], columns=['geometry'], crs=gdf_city_blocks.crs)
    else:
        superblock_merged = gpd.GeoDataFrame(crs=gdf_city_blocks.crs) # empty

    return superblock_merged


def add_attribute_intersection(G, gdf_landuse, label='tags.landu', label_new='landuse'):
    """Intersect edge elements with polygon gdf and assign label if fully within industry landuse
    """
    assert gdf_landuse.index.is_unique
    nx.set_edge_attributes(G, None, label_new)

    rTree = index.Index()
    for index_landuse in gdf_landuse.index:
        geometry = gdf_landuse.loc[index_landuse].geometry
        rTree.insert(index_landuse, geometry.bounds)

    for edge in G.edges:
        edge_geom = G.edges[edge]['geometry']
        landuse_geom_close = list(rTree.intersection(edge_geom.bounds))

        # Get iteratively all nodes to cluster into one new node
        for node_id in landuse_geom_close:
            close_geom = gdf_landuse.loc[node_id].geometry

            if close_geom.intersects(edge_geom) or close_geom.contains(edge_geom):
                G.edges[edge][label_new] = gdf_landuse.loc[node_id][label]

    return G


def get_intersecting_edges(G_input, bb):
    """Get all edges which intersect the bounding box (but not clip)
    """
    G = G_input.copy()
    bb_geom = bb.geometry[0]
    for edge in G.edges:
        edge_geom = G.edges[edge]['geometry']
        if edge_geom.intersects(bb_geom) or bb_geom.contains(edge_geom):
            pass
        else:
            G.remove_edge(edge[0], edge[1])

    return G


def flow_reg_glob_calc(
        G,
        f_local=1,
        f_gobal=2,
        label_one='local_p',
        label_two='glob_p',
        label='flow_ov'
    ):
    """Caclulate overall flow based on local and global flow
    and normalize
    """
    # ---- Calculate by weithing local and global values
    max_value = 0
    nx.set_edge_attributes(G, 0, label)
    for edge in G.edges:

        p_flow_local = G.edges[edge][label_one]
        p_flow_global = G.edges[edge][label_two]

        # Calculate overall flow
        overall_flow = ((p_flow_local * f_local) + (p_flow_global * f_gobal)) / (f_local + f_gobal)

        G.edges[edge][label] = overall_flow

        if overall_flow > max_value:
            max_value = overall_flow

        assert overall_flow <= 1
    '''# -----Normalized the flow
    for edge in G.edges:
        value = G.edges[edge][label]
        norm_value = value / max_value
        G.edges[edge][label] = norm_value'''

    return G


def clip_outer_polygons(gdf, bb):
    """Clip polygons by bb
    """
    gdf = gdf.reset_index(drop=True)
    #print("clip_outer_polygons 734", gdf.shape, bb)

    generic_index = -100
    index_to_remove = []
    for gdf_index in gdf.index:

        polygon = gdf.loc[gdf_index].geometry

    #    print(
    #        "polygon", polygon, "\n",
    #        "bbox", bb, "\n",
    #        "bbox contains polygon", bb.contains(polygon), "\n",
    #        "bbox intersects polygon", bb.intersects(polygon), "\n",
    #    )

        if bb.contains(polygon):
            pass
        elif bb.intersects(polygon):
            intersection = polygon.intersection(bb)
            if intersection.type == 'MultiLineString' or intersection.type == 'MultiPolygon':
                generic_index -= 1
                elements = gdf.loc[gdf_index]
                for geometry_element in intersection:
                    elements['geometry'] = geometry_element
                    for key, value in elements.items():
                        gdf.at[generic_index, key] = value
            else:
                gdf.at[gdf_index, 'geometry'] = intersection
        else:
            index_to_remove.append(gdf_index)

    gdf = gdf.drop(index=index_to_remove)
    gdf = gdf.reset_index(drop=True)

    return gdf


def graph_clip_outer(G, bb):
    """ITerate graph and get line polygon of edge
    and try to clip with polygon. Keep inner

    Note: Needs a have a simplified (not multi-edge) as input
    """
    edges_to_add = []
    edges_to_remove = []
    nodes_to_remove = []

    for edge in G.edges:
        line_geometry = G.edges[edge]['geometry']
        line_attributes = G.edges[edge]

        # Cookie cutter clipping away
        if bb.crosses(line_geometry):
            #clipped_geometry = polygon.difference(merged_pervious)
            #ss = merged_pervious.symmetric_difference(polygon)
            clipped_line = line_geometry.intersection(bb)  # Keep street bit which is in polygon
            coord_one_end = clipped_line.coords[0]
            coord_other_end = clipped_line.coords[-1]

            assert clipped_line.type == 'LineString'

            new_edge = (coord_one_end, coord_other_end)
            line_attributes['geometry'] = clipped_line

            edges_to_add.append((new_edge, line_attributes))
            edges_to_remove.append(edge)

        elif bb.intersects(line_geometry) or bb.contains(line_geometry):
            pass
        else:
            #edges_outside.append(edge) 
            edges_to_remove.append(edge)  # Not sectection, ergo outside
            for node in line_geometry.coords:
                nodes_to_remove.append(node)

    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])

    for edge_index, edge_attributes in edges_to_add:
        G.add_edge(edge_index[0], edge_index[1])
        for attribute_key, attribute in edge_attributes.items():
            G.edges[edge_index][attribute_key] = attribute


    for node in nodes_to_remove:
        try:
            G.remove_node(node)
        except:
            pass

    return G


def check_path_surrounds_node(path, node):
    """test id polygon created out of path contains node or node is on border
    if yes, then the node is within polygon
    """
    #assert path[0] == path[-1]  # closed
    try:
        path_polygonized = Polygon(Point(i) for i in path)
    except: #could not create polygon, thus something is faulty
        return False

    if path_polygonized.contains(Point(node)):
        return True
    else:
        return False


def test_unique_path(path):
    """test if node is twice in a path
    """
    polygon_boundary_length = Polygon(path).boundary.length
    line_path_length = LineString(path).length

    if polygon_boundary_length == line_path_length:
        return True
    else:
        # If length along polygon is not the same, then the path contains
        # douple path elements
        return False


def sort_geographically_closest(list_nodes):
    """always add closest nodes
    """
    list_nodes = copy.copy(list_nodes)
    sorted_nodes = []
    node_from = list_nodes.pop()

    while len(list_nodes) > 0:
        sorted_nodes.append(node_from)

        dist = 99999999
        for node in list_nodes:
            #dist = math.hypot(x2 - x1, y2 - y1)
            dist_n = math.hypot((node_from[0] - node[0])**2 + (node_from[1] - node[1])**2)
            #dist_n = math.dist(node, node_from)
            if dist_n < dist:
                dist = dist_n
                node_closest = node

        list_nodes.remove(node_closest)
        node_from = node_closest

    return sorted_nodes


    
def sort_geographically_clockwise(list_nodes):
    """Sort points clockwise to form polygon
    Taken from: https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
    """
    def centerXY(xylist):
        x, y = zip(*xylist)
        length = len(x)
        return sum(x) / length, sum(y) / length

    cx, cy = centerXY(list_nodes)

    sortedPoints = sorted(list_nodes, key=lambda x: math.atan2((x[1] - cy),(x[0] - cx)))

    return sortedPoints


def sort_geogr_graph(cycle, G):
    """
    """
    first_element = cycle[0]
    nodes_geographically_sorted = [first_element]

    while len(cycle) > 0:
        cycle.remove(first_element)
        for next_element in cycle:
            test_edge = (first_element, next_element)
            if test_edge in G.edges:
                first_element = next_element
                nodes_geographically_sorted.append(next_element)
                break
            else:
                #path needst obe search with shorest path on G
                #shortest_paths = list(nx.all_shortest_paths(G, source=test_edge[0], target=test_edge[1], weight='m_distance'))
                pass

    return nodes_geographically_sorted

'''
def sort_geographically_clockwise(list_nodes, nr_of_entries):
    """Sort points in a list clockwise
    https://dip4fish.blogspot.com/2012/12/simple-direct-quadrilaterals-found-with.html

    """
    if nr_of_entries > 5:
        raise Exception("Warning: sort_geographically_clockwise might take a very long time {}".format(len(nr_of_entries)))
        # Maybe alternatively try the grand cycle functinoality

    all_combinations = list(itertools.permutations(list_nodes, nr_of_entries))

    # Combination with largest area is correct order
    area = 0
    for cnt, combination in enumerate(all_combinations):
        combination_polygon = Polygon(combination)
        if combination_polygon.area > area:
            area = combination_polygon.area
            largest_pos = cnt

    clockwise_sorted = all_combinations[largest_pos]

    return list(clockwise_sorted)
'''


def get_all_edges_in_block(G, polygon):
    """Get all edges which are fully within a polygon"""
    G_within = nx.Graph()
    for edge in G.edges:
        if polygon.contains(Point(edge[0])) and polygon.contains(Point(edge[1])):
            G_within.add_edge(edge[0], edge[1])
            edge_attributes = G.edges[edge]
            for key, value in edge_attributes.items():
                G_within.edges[(edge[0], edge[1])][key] = value

    return G_within
            

def find_grand_cycle(
        G,
        all_neighbours_of_cycle,
        inner_street,
        relaxation_crit=0
    ):
    """Based on list with nodes, find the shortest path which crosses all nodes

    relaxation_crit: 
        Criteria which states for how many nodes it is ok if they are not reached
        Note: IF relaxtion_crit > 0, then also do not consider paths which have
        already been traversed. If == 0, then in each iteration, the path
        calculated along the nodes is deleted.

    Methods
    1. Select random two nodes
    2. Calculate shortest path
    3. Iterate path and check which nodes in list have been already checked
    4. Select next random node --> repeat until no nodes left
    """
    neighbour_nodes = all_neighbours_of_cycle.copy()
    G_for_search = G.copy()
    nr_of_nodes_not_on_path = 0
    path_crit = True
    initial_start_node = neighbour_nodes[-1]  # because pop takes last element
    start_node = neighbour_nodes.pop()
    simple_edges_full_path = []
    path_complex = []
    first_iteration = True
    relax_cnt = 0

    # Add distance to graph #TODO: MAYBE put earlier
    for edge in G_for_search.edges:
        G_for_search.edges[edge]['m_distance'] = G_for_search.edges[edge]['geometry'].length

    while len(neighbour_nodes) > 0:
        next_node = neighbour_nodes.pop()

        # Find shorest path (simple and complex)
        try:
            shortest_paths = list(nx.all_shortest_paths(
                G_for_search, source=start_node, target=next_node, weight='m_distance'))
            shortest_path_simple = shortest_paths[0]
            start_node = next_node

            # --Remove already checked edges from Graph and create simple path
            for node in shortest_path_simple:
                if node in neighbour_nodes:
                    neighbour_nodes.remove(node)
            edges_simple = to_tuple_list(shortest_path_simple)

            # ---Get full edges with intermediary nodes (complex path)
            edge_full_details = []
            for edge in edges_simple:
                simple_edges_full_path.append(edge)
                edge_full_detail = G_for_search.edges[edge]['geometry']
                edges_complex = list(edge_full_detail.coords)
                # Check which end of line is closest and inverst if opposit direction TODO: PUT IN FUNCTION
                edge_full_details = dock_to_path(edge_full_details, edges_complex)

            # Add all coordinates to list (check direction again). Add to node with zero distance
            path_complex = dock_to_path(path_complex, edge_full_details)
        except:
            # no path found
            relax_cnt += 1
            if relax_cnt > relaxation_crit:
                nr_of_nodes_not_on_path += 1
                neighbour_nodes = []
                path_crit = False
                continue  #return to top
            else: # node cannot be reached, therfore take next
                start_node = next_node #path[-1]  # change to last node on path
                continue

        # Because otherwise same path could be calculated backwards. Remove from search
        # But only delete if not relax criteria is activated
        if not relaxation_crit:
            G_for_search.remove_edges_from(edges_simple)

        # add last node to close loop and have last edge as well
        if first_iteration:
            neighbour_nodes.insert(0, initial_start_node)  
            first_iteration = False
    try:
        block_cycling = Polygon(path_complex)
        
        if not block_cycling.is_valid:
            block_cycling = block_cycling.buffer(0)

        block_cycling_area = block_cycling.area

        # Check whether entire inner street is contained (or on line) of path
        all_inner_street_contained = True
        for innernode in inner_street:
            if block_cycling.touches(Point(innernode)) or block_cycling.contains(Point(innernode)):
                pass
            else:
                all_inner_street_contained = False
                break
        
        if not all_inner_street_contained:
            block_cycling_area = 0
            path_crit = 0
            block_cycling = gpd.GeoDataFrame()
        else:
            pass
    except:
        block_cycling_area = 0
        path_crit = 0
        block_cycling = gpd.GeoDataFrame()

    # Calculate path length
    if path_crit:
        path_length = get_distance_along_path(G, simple_edges_full_path, crit_edges=True)
    else:
        path_length = 0
    
    # Check if path crosses a bridge (then ignore it) # New 13.12.2021
    for edge in simple_edges_full_path:
        tunnel_crit = G.edges[edge]['tags.tunnel']
        bridge_crit = G.edges[edge]['tags.bridge']
        if tunnel_crit == 1 or bridge_crit == 1:
            path_crit = False
   
    return path_crit, path_length, block_cycling_area, block_cycling


def remove_edge_by_attribute(
        G,
        attribute='tags.highway',
        value="service"
    ):
    """Remove edge by attribute
    """
    for edge in G.edges:
        if G.edges[edge][attribute] == value:
            G.remove_edge(edge[0], edge[1])

    return G


def calc_eccentricity(G, label):
    """Calculate eccentricity measure for lagest subgraph
    """
    nx.set_edge_attributes(G, 0, name=label)

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_subgraph = G.subgraph(Gcc[0])
    G_eccentricity = nx.eccentricity(largest_subgraph)  # Works only if fuly connected graph

    for node, value in G_eccentricity.items():
        G.nodes[node][label] = round(value, 4)

    return G


def get_length_network(G):
    """Get length of all edges in a network
    """
    distance = 0
    for edge in G.edges:
        distance += G.edges[edge]['geometry'].length

    return distance


def get_distance_along_path(G, path, crit_edges=False):
    """Iterat graph and get distance from geometry attribute
    crit_edges : True --> Takes as input tuple edges and not a path
    """
    distance = 0
    if crit_edges:
        edges = path
    else:
        edges = to_tuple_list(path)
    for edge in edges:
        distance += G.edges[edge]['geometry'].length

    return distance


def calc_min_circle_distance(G):
    """Calculate the minimum distance it takes
    to walk through the graph with a breath serach until
    one reaches full circle
    """
    # Add distance to graph
    for edge in G.edges:
        G.edges[edge]['m_distance'] = G.edges[edge]['geometry'].length

    print("... calculating shortest potential circle path for each node")
    for node in G.nodes:
        if G.degree[node] > 2:
            # Get neighbors and try to find loop.
            nodes_neighbouring = list(G.neighbors(node)) # neighbors() and successors() are the same.

            # Search shortest path from neighbouring nodes
            distance = 999999
            found_loop = False
            for neigbor in nodes_neighbouring:
                edge_attributes = G.edges[(neigbor, node)]
                length_fist_edge_not_in_path = edge_attributes['geometry'].length
                G.remove_edge(neigbor, node) # temporally remove edge to calculate path

                try:
                    shortest_paths = list(nx.all_shortest_paths(G, source=neigbor, target=node, weight='m_distance'))
                except Exception as e:
                    
                    #print(e) # no path hfound
                    # Add edge again
                    G.add_edge(neigbor, node)
                    for key, value in edge_attributes.items():
                        G.edges[(neigbor, node)][key] = value
                    continue

                # Add edge again                                
                G.add_edge(neigbor, node)
                for key, value in edge_attributes.items():
                    G.edges[(neigbor, node)][key] = value

                shortest_path = shortest_paths[0]  # hopefully only one single path
                distance_path = get_distance_along_path(G, shortest_path)
                distance_path += length_fist_edge_not_in_path
                if distance_path < distance:
                    distance = distance_path
                    found_loop = True

            if found_loop:
                G.nodes[node]['c_dist'] = round(distance, 2)
            else:
                G.nodes[node]['c_dist'] = 0
        else:
            G.nodes[node]['c_dist'] = 0  # no loop bec

    # Remove attribute
    for node_from, node_to, d in G.edges(data=True):
        d.pop('m_distance')

    return G


def load_centrality(G, label='cen_lity'):
    """Centrality
    """
    G_undirected = nx.to_undirected(G)
    a = nx.load_centrality(G_undirected, normalized=True)
    for node, value in a.items():
        G.nodes[node][label] = round(value, 4)

    G = calculate_average_edge_attribute_from_node(G, label=label)

    return G


def betweeness_centrality(G, label='btw'):
    """
    https://coderzcolumn.com/tutorials/data-science/network-analysis-in-python-node-importance-and-paths-networkx#5.5
    """

    G_undirected = nx.to_undirected(G)
    a = nx.betweenness_centrality(G_undirected, normalized=False)
    for node, value in a.items():
        G.nodes[node][label] = value
    G = calculate_average_edge_attribute_from_node(G, label='btw')

    a = nx.betweenness_centrality(G_undirected, normalized=True)
    for node, value in a.items():
        G.nodes[node]['{}_norm'.format(label)] = round(value, 4)

    return G


def calc_edge_connectivity(G, label='loc_conn'):
    """
    Local edge connectivity for two nodes s and t is the 
    minimum number of edges that must be removed to disconnect them.
    """
    print("...calculating edge connectivity")
    for edge in G.edges:
        G.edges[edge][label] = local_edge_connectivity(G, edge[0], edge[1])
    print("...finished calculating edge connectivity")

    return G


def calc_vitality(G):
    """
    Closeness vitality of a node is the change in the sum of distances
    between all node pairs when that node is removed from the network.
    NOTE: SUBCOMPONENT MAY BE THE INTERESTING BIT
    """
    to_directed = G.to_directed()
    #connected_subgraphs = list(nx.strongly_connected_components(to_directed))
    
    # Get largest connected subgraph, i.e. connected component
    #https://stackoverflow.com/questions/26105764/how-do-i-get-the-giant-component-of-a-networkx-graph
    subgraph_list = sorted(nx.connected_components(G), key=len, reverse=True)
    list_largest_subgraph = subgraph_list[0]
    strongly_connected_subgraph = G.subgraph(list_largest_subgraph)
    print("...calculating vitality")
    nodes_determined = {}
    for paths_nodes in list_largest_subgraph:
        for node in paths_nodes:
            clossenss_vitality = nx.closeness_vitality(strongly_connected_subgraph, node)
            nodes_determined[node] = clossenss_vitality

    for node, clossenss_vitality in nodes_determined.items():
        G.nodes[node]['vitality'] = clossenss_vitality
    print("finished calculating vitality")

    return G


def calculate_closeness_centrality(G):
    """
    """
    a = nx.algorithms.centrality.closeness_centrality(G)
    for node, value in a.items():
        G.nodes[node]['closeness_centrality'] = value

    return G


def find_cycles(G):
    """
    
    https://www.geeksforgeeks.org/detect-cycle-undirected-graph/
    
    """
    #TODO: EXPLORE https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.find_cycle.html

    cycles = nx.find_cycle(G, orientation='ignore') #n="original")
    #cycles = [i[:-1] for i in list(cycles)] # Remove orientation

    # Works for undirected graphs
    cycle_cylce_basis = nx.cycle_basis(G)

    # Simple cycles (only three node cycles)
    #simple_cycles = nx.simple_cycles(G)

    # Works only for directed graphs
    #cycles = nx.simple_cycles(G_own) # https://stackoverflow.com/questions/35683302/python-networkx-detecting-loops-circles
    #v = hp_net.to_tuple_list(cycle)


    # --- Add cycle edges to directed Graph
    # https://stackoverflow.com/questions/12367801/finding-all-cycles-in-undirected-graphs
    nx.set_edge_attributes(G, 0, "cycle_B")
    nx.set_edge_attributes(G, 0, "cycle_A")

    for cycle in cycle_cylce_basis:
        cycle_edges = to_tuple_list(cycle)
        for edge in cycle_edges:
            try:
                G.edges[edge]['cycle_A'] = 1
            except KeyError:
                G.edges[(edge[1], edge[0])]['cycle_A'] = 1 

    for cycle in cycles:
        cycle_edge = (cycle[0], cycle[1])
        try:
            G.edges[cycle_edge]['cycle_B'] = 1
        except KeyError:
            G.edges[(cycle_edge[1], cycle_edge[0])]['cycle_B'] = 1

    return G


def calc_strahler_simplified(G, depth_limit=10, label='strahler'):
    """Calculate simplified strahler
    
    Take all nodes with 1 neighbour and then do a depth, search with maximum depth
    """
    nx.set_node_attributes(G, 0, "strahler")
    if nx.is_directed(G):
        nr_edges_line_end = 1
    else:
        nr_edges_line_end = 2

    for node_index in G.nodes:
        node = G.nodes[node_index]
        degree = G.degree[node_index]

        if degree == nr_edges_line_end:
            strahler_nr = 1
            G.nodes[node_index][label] = strahler_nr
            
            # Depth first search
            traveral_edges = list(nx.dfs_edges(G, source=node_index, depth_limit=depth_limit))

            while len(traveral_edges) > 0:
                traveral_edge = traveral_edges.pop(0)  # first element
                from_node = traveral_edge[0]
                to_node = traveral_edge[1]
                G.nodes[to_node][label] += 1
       
    return G


def calc_orentation(G):
    """Calculate absolute orientation of edge based on start and end node
    """
    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180°)'''
        angle = math.atan2(point2.x - point1.x, point2.y - point1.y)
        if angle > 0:
            abs_angle = math.degrees(angle)
        else:
            abs_angle = math.degrees(angle) + 180
        return abs_angle

    nx.set_edge_attributes(G, 0, 'azimut')

    for edge in G.edges:
        start_node = Point(edge[0])
        end_node = Point(edge[1])

        azimut_value = azimuth(start_node, end_node)

        G.edges[edge]['azimut'] = azimut_value

    return G


def block_node_types(G):
    """
    input: undirected
    Type A: Barcelon 3x3 superblocck

    Type B: Miniblock 2x2

    """
    #
    nx.set_edge_attributes(G, 0, 'potential_block')
    #nx.set_edge_attributes(G, 0, 'get_upstream_numbers')

    for node_index in G.nodes:
        node = G.nodes[node_index]

        # Number of neigbors
        degree = G.degree[node_index]

        # Get all neighbouring nodes
        if degree > 4:
            G.nodes[node_index]['potential_block'] = 1

        # Direct neighbors which are not on same road
        #negibors = list(G.neigbhors(node_index))
        
        # Make depth search and get upstrean number of edges
        #a = dict(nx.bfs_predecessors(G, source=node_index, depth_limit=2))
        #G.nodes[node_index]['get_upstream_numbers'] = 
        
    # Test if neighbouring are also potentials blocks
    nx.set_edge_attributes(G, 0, 'nr_neighbour_pot_block')

    # --------------------
    # Get miniblock
    # --------------------
    for node_index in G.nodes:
        node = G.nodes[node_index]

        # Number of neigbors
        potential_block = G.nodes[node_index]['potential_block']

        #if potential_block:
            
        #    # 
    
    
    nr_of_nodes = 0
    max_search_distance = 200  # [m]
    graph_size_to_ignore = 8  # [nr of nodes] If more nodes, this node is considered as a temp node

    for node_index in G.nodes:
        node = G.nodes[node_index]

        # Get all 
        # Get all nodes along (depth-search)
        #nx.dfs_edges(G[, source, depth_limit])
    

        G.nodes[node_index]['nr_neighbour_pot_block'] = nr_of_nodes
        # Get for each ndoe, the number of block nodes

    return G

def geometric_simplification(G, max_distance=20):
    """
    1. Get all clusters to nodes which need to be merged into one
    and calculate new graph node 
    """
    # Create node tree
    rTree = index.Index()
    for node_iloc, index_xy in enumerate(G.nodes):
        node = Point(index_xy)
        rTree.insert(node_iloc, node.bounds)

    # Merge nodes and simpfily network
    G_new = nx.Graph()
    checked_nodes = []
    nodes_to_remove = []
    reassigned_nodes = {}
    bar = Bar('Geometric simplification:', max=G.number_of_nodes())

    for node_nr in G.nodes:
        node = G.nodes[node_nr]

        # Only if node is at an intersection and has not been already checked
        if (node['degree'] > 2) and (node_nr not in checked_nodes):
            nodes_to_cluster = []
            x_coor = []
            y_coor = []
            cluster_nodes_to_check = [node_nr]

            while len(cluster_nodes_to_check):
                node_nr = cluster_nodes_to_check.pop()
                node_buffer = Point(node_nr).buffer(max_distance)
                nodes_closes = list(rTree.intersection(node_buffer.bounds))

                # Get iteratively all nodes to cluster into one new node
                for node_ind in nodes_closes:
                    node_close_index = list(G.nodes)[node_ind]  # Note get list element

                    if G.nodes[node_close_index]['degree'] > 2:  # Only try to merge if also a intersecting node
                        node_close_index_geom = Point(node_close_index)
                        # If spatially contains and not self
                        if (node_close_index_geom.within(node_buffer)) and (node_close_index != node_nr): 
                            if node_close_index not in checked_nodes:  # Check if not already checked
                                nodes_to_cluster.append(node_close_index)
                                x_coor.append(node_close_index_geom.x)
                                y_coor.append(node_close_index_geom.y)
                                checked_nodes.append(node_close_index)
                                cluster_nodes_to_check.append(node_close_index)
                                nodes_to_remove.append(node_close_index)               

            # Calculate new position of average cluster node
            mean_node = (np.mean(x_coor), np.mean(y_coor))
            G_new.add_node((mean_node[0], mean_node[1]))

            # Reassignement dict
            for node in nodes_to_cluster:
                reassigned_nodes[node] = mean_node  

            # I. Get all neighbors of all the nodes to cluster 
            # II. Add new edges to new graph
            # III. Delete all edges in old graph
            for node_close in nodes_to_cluster:
                neighbor_nodes = list(G.neighbors(node_close))  # Get neighbors in old graph
                
                # Check if in new network #NOTE: IS THIS REALLY NEEDED?
                if node_close in G_new.nodes():
                    neighbor_nodes_old_graph = list(G_new.neighbors(node_close))
                    neighbor_nodes = neighbor_nodes + neighbor_nodes_old_graph

                for neigbor in neighbor_nodes:
                    if neigbor not in nodes_to_cluster:

                        # Check if neigbour was already reassigned
                        if neigbor in reassigned_nodes:
                            neigbor = reassigned_nodes[neigbor]  # Switch ID
                            edge_attributes = G_new.edges[(neigbor, node_close)]  # Get geomegry from new node
                            
                            # NEW: ADDED HERE MAYBE NEEDED
                            ##G_new.remove_edge(neigbor, node_close)  #  Remove in new graph
                            #try:
                            #    G_new.remove_edge(neigbor, node_close) # if already ne connection established
                            #except:
                            #    pass
                        else:
                            # Get attributes, including geometry
                            if (neigbor, node_close) not in G_new.edges() and (neigbor, node_close) not in G.edges():
                                continue  # Not in new or old
                            elif (neigbor, node_close) in G_new.edges():
                                edge_attributes = G_new.edges[(neigbor, node_close)]
                                G_new.remove_edge(neigbor, node_close)  #  Remove in new graph
                            else:
                                edge_attributes = G.edges[(neigbor, node_close)]
                                G.remove_edge(neigbor, node_close)  #  Remove in old graph
                                #edge_attributes = G.edges[(node_close, neigbor)]
                                #G.remove_edge(node_close, neigbor)  #  Remove in old graph

                        # Adapt geometry that last edge is in line
                        if edge_attributes['geometry'].coords[0] == neigbor:
                            all_pnts = [i for i in edge_attributes['geometry'].coords]
                        else:
                            all_pnts = [i for i in edge_attributes['geometry'].coords[::-1]]
                        all_pnts.pop()  # remove last connecting node
                        all_pnts.append(mean_node)
                        new_line_geometry = LineString(all_pnts)

                        assert new_line_geometry.type == 'LineString'
                        edge_attributes['geometry'] = new_line_geometry

                        # Add new edge to new graph
                        G_new.add_edge(neigbor, mean_node)
                        for val, attr in edge_attributes.items():
                            G_new.edges[(neigbor, mean_node)][val] = attr

            # Remove all edge combination between cluster nodes
            all_nodes_to_cluster_combinations = itertools.combinations(nodes_to_cluster, 2)
            for combination in all_nodes_to_cluster_combinations:
                try:
                    G.remove_edge(combination[0], combination[1])
                except:
                    pass  # no connection in network
        bar.next()
    bar.finish()

    # Remove nodes
    G.remove_nodes_from(nodes_to_remove)

    # Compose
    G = nx.compose(G, G_new)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def clean_simple_intersections(G_roads, single_segments=True):
    """Check each edge whether spatial intersection. If yes, add intersection point
    
    """

'''
def merge_close_nodes(
        G,
        max_distance=20
    ):
    """
    1. Get all nodes which are candidates to merge
        -> nodes with degree > 2
        -> nodes which are close within a certain distance
    """
    checked_nodes = []

    rTree = index.Index()
    for node_iloc, index_xy in enumerate(G.nodes):
        node = Point(index_xy)
        rTree.insert(node_iloc, node.bounds)

    bar = Bar('Testing if nodes can be merged:', max=G.number_of_nodes())
    for node_nr in G.nodes:
        node = G.nodes[node_nr]
        if node['degree'] > 2: # Only if node is at an intersection
            node_buffer = Point(node_nr).buffer(max_distance)
            nodes_closes = list(rTree.intersection(node_buffer.bounds))

            for node_ind in nodes_closes:

                node_close_index = list(G.nodes)[node_ind]
                if G.nodes[node_close_index]['degree'] > 2: # Only try to merge if also a intersecting node
                    node_close_index_geom = Point(node_close_index)
                    if node_close_index_geom.within(node_buffer) and node_close_index != node_nr: # not self

                        # Check if not already checked
                        if node_close_index not in checked_nodes:

                            # Remove original node 
                            try:               
                                G.remove_edge(node_nr, node_close_index)
                            except:
                                pass # there was no edge between the points

                            # Reassign edges of neigbors and delete edges
                            neighbor_nodes = list(G.neighbors(node_close_index))

                            for neigbor in neighbor_nodes:
                                edge_attributes = G.edges[(neigbor, node_close_index)]
                                #old_line = edge_attributes['geometry'].coords
                                #old_line = [i for i in edge_attributes['geometry'].coords]
                                #if old_line[:-1] == neigbor:
                                #    old_line = old_line[::-1]  # Reverse line

                                #new_line_coordinates = old_line[:-1]
                                #new_line_coordinates.append(node_nr)
                                #new_line_geometry = LineString([i for i in new_line_coordinates])
                                new_line_geometry = LineString((neigbor, node_nr))
                                #    print("...")
                                assert new_line_geometry.type == 'LineString'

                                edge_attributes['geometry'] = new_line_geometry

                                # Remove old edge and add new one
                                G.remove_edge(neigbor, node_close_index)

                                # Replace  last point
                                G.add_edge(neigbor, node_nr)
                                for val, attr in edge_attributes.items():
                                    G.edges[(neigbor, node_nr)][val] = attr

                            checked_nodes.append(node_close_index)
        bar.next()
    bar.finish()

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
'''
def calculate_average_edge_attribute_from_node(G, label='btw_norm'):
    """Calculate average value per edge
    """
    for edge in G.edges:
        start_node = G.edges[edge]['geometry'].coords[0]
        end_node = G.edges[edge]['geometry'].coords[-1]
        density_start_node = G.nodes[start_node][label]
        density_end_node = G.nodes[end_node][label]
        G.edges[edge][label] = (density_start_node + density_end_node) / 2

    return G


def calculate_node_degree(G):
    """If directed, then the number of edges is
    always double and thus needs to be divided by 2
    """
    directed_crit = nx.is_directed(G)

    for node, degree in G.degree:
        if directed_crit:
            G.nodes[node]['degree'] = degree / 2
        else:
            G.nodes[node]['degree'] = degree

    return G
    
'''def marshall_depth(G, tag_datum='tags.highway', tag_crit_daum='primary'):
    """
    Calculate distance from a "datum" (i.e. major roads)

    1. Get all datum roads, assign depth --> assign 1

    2. Get all edges ending in datum roads --> assign 2

    Breath serach until network is done

    """
    nx.set_edge_attributes(G, 0, "marshall_depth")

    assignes_edges = []

    # (1) Set datum depth
    for edge_index in G.edges:
        edge_crit = G.edges[edge_index][tag_datum]
        if edge_crit == tag_crit_daum:
            G.edges[edge_index]["marshall_depth"] = 1
            assignes_edges.append(edge_index)

    # (2) Iterative depth assignment based on breath search
    
    for nx.dfs_edges(G[, source, depth_limit]):
    
    return G'''


'''def helper_MultiLineString_to_LineString(x, y):
    """
    """
    geometry_list = []
    cnt = 0
    for x_2, y_2 in zip(x, y):
        pnt_2 = (x_2, y_2)
        if cnt > 0:
            geometry_list.append(LineString((pnt_1, pnt_2)))
        pnt_1 = (x_2, y_2)
        cnt += 1
    return geometry_list'''



def remove_trinagle(G, max_distance_of_triangle=30):
    """https://stackoverflow.com/questions/1705824/finding-cycle-of-3-nodes-or-triangles-in-a-graph

    Find all trinales with a certain circumference and remove them
    """
    G_undicrected = G.to_undirected()

    print("Number of triangles: {}".format(len(nx.triangles(G_undicrected))))
    print("Number of edges:     {}".format(G.number_of_edges()))
    all_cliques = nx.enumerate_all_cliques(G_undicrected)

    triad_cliques = [x for x in all_cliques if len(x) == 3]

    edges_to_remove = []
    for triangle in triad_cliques:
        edge_1 = G_undicrected.edges[triangle[0], triangle[1]]['geometry']
        edge_2 = G_undicrected.edges[triangle[1], triangle[2]]['geometry']
        edge_3 = G_undicrected.edges[triangle[2], triangle[0]]['geometry']
        triangle_edges = [edge_1, edge_2, edge_3]
        triangle_lenghts = [i.length for i in triangle_edges]
        trinagle_length = sum(triangle_lenghts)

        if trinagle_length <= max_distance_of_triangle:
            smallest_edge_position = triangle_lenghts.index(min(triangle_lenghts))
            edges_to_remove.append(triangle_edges[smallest_edge_position])

    if len(edges_to_remove) > 0:
        for edge in edges_to_remove:
            try:
                # Because directed, needs to be deleted both ways
                G.remove_edge(edge.coords[0], edge.coords[1])
                G.remove_edge(edge.coords[1], edge.coords[0])
            except: 
                pass # already removed

    print("Number of triangle nodes with max circumference: {}".format(len(edges_to_remove)))
    print("Number of edges: {}".format(G.number_of_edges()))

    return G


def remove_intersect_building(G, buildings):
    """Remove all edges which intersect buildings
    and are not a major road
    """
    print("... remove_intersect_building")
    ignore_attributes = [
        'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link']
    
    rTree = index.Index()
    for edge_nr, building in enumerate(buildings.geometry):
        rTree.insert(edge_nr, building.bounds)

    # Remove all edges which intersect a building
    edges_to_drop = []
    for edge in G.edges:
        edge_geom = G.edges[edge]['geometry']
        
        if G.edges[edge]['tags.highway'] in ignore_attributes:
            continue
        else:
            buildings_intersection = list(rTree.intersection(edge_geom.bounds))
            for building_intersection in buildings_intersection:
                build_geom = buildings.iloc[building_intersection].geometry
                if edge_geom.intersects(build_geom):
                    edges_to_drop.append(edge)

    if len(edges_to_drop) > 0:
        G.remove_edges_from(edges_to_drop)

    # Remove isolated nodes 
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
    
    '''nodes_to_check = []
    for node in G.nodes:
        if G.nodes[node]['degree'] == 1:
            nodes_to_check.append(node)

    rTree = index.Index()
    for edge_nr, building in enumerate(buildings.geometry):
        rTree.insert(edge_nr, building.bounds)

    nodes_to_drop = []
    for node in nodes_to_check:
        buildings_intersection = list(rTree.intersection(Point(node).bounds))

        for building_intersection in buildings_intersection:
            build_geom = buildings.iloc[[building_intersection]].geometry
            if list(build_geom.contains(Point(node)))[0]:
                nodes_to_drop.append(node)
                break

    # Remove all edges which intersect a building
    edges_to_drop = []
    for edge in G.edges:
        #edge_geom = json.loads(G.edges[edge]['Json'])
        edge_geom = G.edges[edge]['geometry']
        buildings_intersection = list(rTree.intersection(edge_geom.bounds))

        for building_intersection in buildings_intersection:
            build_geom = buildings.iloc[building_intersection].geometry
            if edge_geom.intersects(build_geom):
                edges_to_drop.append(edge)

    if len(nodes_to_drop) > 0:
        G.remove_nodes_from(nodes_to_drop)
    
    if len(edges_to_drop) > 0:
        G.remove_edges_from(edges_to_drop)

    # Remove isolated nodes 
    G.remove_nodes_from(list(nx.isolates(G)))

    return G    '''


def remove_zufahrt(G, max_length=5):
    """Remove nodes with degree 1 (final nodes) which are
    very small and thus possible a fine connecting road to a 
    house or similar
    """
    nodes_to_drop = []
    for node in G.nodes:
        calculated_degree = G.nodes[node]['degree']
        if calculated_degree == 1:
            neighbor_list = list(G.neighbors(node)) # neighbors() and successors() are the same.
            if len(neighbor_list) == 0:
                single_neigbor = list(G.predecessors(node))[0]
                length_edge = G.edges[(single_neigbor, node)]['geometry'].length
            else:
                single_neigbor = neighbor_list[0]
                length_edge = G.edges[(node, single_neigbor)]['geometry'].length

            if length_edge < max_length:
                nodes_to_drop.append(node)                
    if len(nodes_to_drop) > 0:
        G.remove_nodes_from(nodes_to_drop)
    else:
        pass

    return G

def get_line_closest_to_street(clipped_line, line_segment):
    """Get line which is closest to perpendicular line
    or the pnt in the middle of the street. This may happend
    because if clipped with buildings, the perpendicular line may overshoot
    the buildings
    """
    line_touching = None
    split_pnts = [i for i in line_segment.coords]
    for line in clipped_line:
        touching = False
        for coordinate in line.coords:
            if tuple(coordinate) in split_pnts:
                touching = True
        if touching:
            break
    if touching:
        return line
    else:
        return None
        #raise Exception("Line not touching")
    
    

def create_artifical_street_area(gdf_edges_superblock, buildings):
    """Create artificial street areas

    1. Split line along individual points with a certain distance
    2. Create receangular lines and clip with buildings and calculate average width of street
    3. Create left and right sided buffer based on average distance
    4. Buffer buildigns and intersectp street space
    """
    min_distance_to_building = 0.5  # [m]
    min_element_length = 1          # [m]
    offset_distance = 30            # [m]
    absolute_max_street_size = 10   # [m]
    absolute_min_street_size = 2    # [m]

    # All merged buildings
    all_buildings_merged = unary_union(buildings.geometry.tolist())
    all_build_buffered = all_buildings_merged.buffer(min_distance_to_building)

    # Search tree for buildings
    rTree = index.Index()
    for counter, centroid in enumerate(buildings.geometry.centroid):
        rTree.insert(counter, centroid.bounds)

    bar = Bar('Street area generation:', max=gdf_edges_superblock.shape[0])
    full_buffers = []
    for edge_index in gdf_edges_superblock.index:

        # Edge geometry
        full_edge = gdf_edges_superblock.loc[edge_index].geometry
        buffer_distance_to_select_buildings = 300
        # Get closest buildings within 300 meter of edge
        buildings_intersection = list(rTree.intersection(full_edge.buffer(buffer_distance_to_select_buildings).bounds))
        buildings_intersection_union_no_buffer = unary_union(buildings.iloc[buildings_intersection].geometry.tolist())
        buildings_intersection_union = buildings_intersection_union_no_buffer.buffer(min_distance_to_building)

        # ----------------------------------------------------------
        # (1) Create split points
        # ----------------------------------------------------------
        edge_length = full_edge.length
        nr_of_spli_ptns = int(edge_length / min_element_length)

        if nr_of_spli_ptns > 0:
            # Spit edge
            split_pnts = [full_edge.coords[0], full_edge.coords[-1]]
            for i in range(1, nr_of_spli_ptns):
                dist = i / nr_of_spli_ptns
                pnt = full_edge.interpolate(dist, normalized=True)
                split_pnts.append((pnt.x, pnt.y))

            split_segments = []
            cnt = 0
            for split_pnt in split_pnts:
                if cnt > 0:
                    split_segments.append(LineString((split_pnt_prev, split_pnt)))
                cnt += 1
                split_pnt_prev = split_pnt

            # ----------------------------------------------------------
            # (2) Create perpendiclar lines and split with buidings and calculate averge width of street
            # ----------------------------------------------------------
            right_length = []
            left_length = []
            for split_segment in split_segments:
                left = split_segment.parallel_offset(offset_distance, 'left')
                right = split_segment.parallel_offset(offset_distance, 'right')
                first_left_line = LineString([left.boundary[0], split_segment.coords[0]])
                second_left_line = LineString([left.boundary[1], split_segment.coords[1]])
                first_right_line = LineString([right.boundary[0], split_segment.coords[1]])
                second_right_line = LineString([right.boundary[1], split_segment.coords[0]])

                # Clip with buildings
                left_lines_to_check = [first_left_line, second_left_line]
                right_lines_to_check = [first_right_line, second_right_line]
                for line_to_check in left_lines_to_check:
                    if line_to_check.intersects(buildings_intersection_union):
                        clipped_line = line_to_check.difference(buildings_intersection_union)                        
                        if clipped_line.type == 'MultiLineString':
                            clipped_line = get_line_closest_to_street(clipped_line, split_segment)
                        if clipped_line != None and clipped_line.length != offset_distance:
                            left_length.append(clipped_line.length)
                for line_to_check in right_lines_to_check:
                    if line_to_check.intersects(buildings_intersection_union):
                        clipped_line = line_to_check.difference(buildings_intersection_union)
                        if clipped_line.type == 'MultiLineString':
                            clipped_line = get_line_closest_to_street(clipped_line, split_segment)
                        if clipped_line != None and clipped_line.length != offset_distance:
                            right_length.append(clipped_line.length)

            if left_length != []:
                #left_buffer_m = np.average(left_length)
                left_buffer_m = np.min(left_length)
            else:
                if right_length != []:
                    #left_buffer_m = np.average(right_length)
                    left_buffer_m = np.min(right_length)
                else:
                    left_buffer_m = 0
            if right_length != []:
                #right_buffer_m = np.average(right_length)
                right_buffer_m = np.min(right_length)
            else:
                if left_length != []:
                    #right_buffer_m = np.average(left_length)
                    right_buffer_m = np.min(left_length)
                else:
                    right_buffer_m = 0

            # Select minimum buffer or left and wright
            min_buffer_both = min(right_buffer_m, left_buffer_m)
            
            # Limit absolute width
            if min_buffer_both > absolute_max_street_size:
                min_buffer_both = absolute_max_street_size
            elif min_buffer_both < absolute_min_street_size:
                min_buffer_both = absolute_min_street_size
            else:
                pass
            right_buffer_m = min_buffer_both
            left_buffer_m = min_buffer_both
            
            # Because sometimes problems with offset parallel line, use iterativevely to change buffer distance
            successful_offset_left = False
            successful_offset_right = False
            offset_iter_m = 0
            while not successful_offset_left:
                parallel_line_left = full_edge.parallel_offset(left_buffer_m + offset_iter_m, side='left', join_style=2)#, mitre_limit=1)
                if parallel_line_left.type == 'LineString':
                    successful_offset_left = True
                else:
                    offset_iter_m += 0.1
            while not successful_offset_right:
                parallel_line_right = full_edge.parallel_offset(right_buffer_m + offset_iter_m, side='right', join_style=2) #, mitre_limit=1)
                if parallel_line_right.type == 'LineString':
                    successful_offset_right = True
                else:
                    offset_iter_m += 0.1

            assert parallel_line_left.type == 'LineString'
            assert parallel_line_right.type == 'LineString'

            # ----------------------------------------------------------
            # (3) Create left and right buffer polygon with help of parallel lines
            # ----------------------------------------------------------
            pointList = []
            for pnt in [Point(i) for i in parallel_line_right.coords]:
                pointList.append(pnt)
            for pnt in [Point(i) for i in full_edge.coords]:
                pointList.append(pnt)
            right_buffer = Polygon(pointList)

            pointList = []
            for pnt in [Point(i) for i in parallel_line_left.coords]:
                pointList.append(pnt)
            pointList = pointList[::-1]
            for pnt in [Point(i) for i in full_edge.coords]:
                pointList.append(pnt)
            left_buffer = Polygon(pointList)

            full_buffer = unary_union([right_buffer, left_buffer])
            full_buffers.append(full_buffer)

    # Becaues some of the buffesr may intersect, remove overlapping parts
    gdf_street_areas = unary_union(full_buffers)
    sliver_dist = 0.45
    gdf_street_areas_polygon = gdf_street_areas.buffer(-1 * sliver_dist).buffer(sliver_dist)
    gdf_street_areas_polygon = gdf_street_areas_polygon.buffer( sliver_dist).buffer(-1 * sliver_dist)
    
     # If multiple polygons, only select largest polygon
    if gdf_street_areas_polygon.type == 'MultiPolygon':
        min_area = 0
        for polygon in gdf_street_areas_polygon.geoms:
            if polygon.area > min_area:
                largest_polygon = polygon
        gdf_street_areas_polygon = largest_polygon 

    gdf_street_areas = gpd.GeoDataFrame([gdf_street_areas_polygon], columns=['geometry'], crs=gdf_edges_superblock.crs)

    # ----------------------------------------------------------
    # (4) Clip street space with buffered buildings
    # ----------------------------------------------------------
    gdf_street_areas['geometry'] = gdf_street_areas.difference(all_build_buffered)
    gdf_street_areas = gdf_multipolygon_to_polygon(gdf_street_areas)

    return gdf_street_areas


def undirected_to_directed(G, label='tags.one'):
    """Convert undirected to directed"""

    G_directed = nx.DiGraph()

    for edge in G.edges:
        label_crit = G.edges[edge][label]

        G_directed.add_edge(edge[0], edge[1])
        for i, j in G.edges[edge].items():
            G_directed.edges[edge[0], edge[1]][i] = j

        if label_crit == 'yes':
            pass
        else:
            G_directed.add_edge(edge[1], edge[0])
            for i, j in G.edges[edge].items():
                G_directed.edges[edge[1], edge[0]][i] = j

    return G_directed  

def remove_rings(gdf):
    """remove rings in gdf
    """
    index_to_drop = []
    for index in gdf.index:
        geometry = gdf.loc[index].geometry

        if geometry.is_ring:
            index_to_drop.append(index)
    print("Number of rings which are removed: {}".format(len(index_to_drop)))
    gdf = gdf.drop(index=index_to_drop)
    gdf = gdf.reset_index(drop=True)
    
    return gdf

def simplify_network(G, crit_big_roads=True, crit_bus_is_big_street=False):
    """Agglomerate all edges which are between two nodes
    which have neighbours into a SingleLine with multiple coordinates
    Note: Works on graph --> if Digraph, then do not simplify those loop-elements
            Loop elements are not simplified fully, as otherwise one is deleted
            if not a multiGraph
    Note: Depends on having an updated version of the connected degreeout_edges
    Note: Nodes on big roads (tram, bus, trolley are not simplified)
    """
    print("... simplifying network")
    assert 'crs' in G.graph.keys()
    crs_from = G.graph['crs']

    G_new_edges = nx.Graph()
    G_new_edges.graph = G.graph

    # Convert to undirected (if directed, use oneway attribute to reset)
    if_directed_input = False
    directed_crit = G.is_directed()
    if directed_crit:
        G = G.to_undirected()
        if_directed_input = True

    nodes_to_remove = []

    edge_nr_to_consider = 2
    for node, degree in G.degree:
        if degree == edge_nr_to_consider:
            edges_connecting = G.edges(node)
            assert len(edges_connecting) == edge_nr_to_consider
    
    bar = Bar('Non geometric simplification: ', max=G.number_of_nodes())
    for node, degree in G.degree:
        if degree == edge_nr_to_consider:
            if directed_crit:
                incoming_edges = list(G.in_edges(node))
                outcoming_edges = list(G.out_edges(node))
                assert len(incoming_edges) + len(outcoming_edges) == edge_nr_to_consider

                if len(incoming_edges) == edge_nr_to_consider:
                    incoming_edge = incoming_edges[0]
                    outcoming_edge = incoming_edges[0]
                elif len(outcoming_edges) == edge_nr_to_consider:
                    incoming_edge = outcoming_edges[0]
                    outcoming_edge = outcoming_edges[0]
                else:
                    incoming_edge = incoming_edges[0]
                    outcoming_edge = outcoming_edges[0]
            else:
                # Get incoming node and outcoming node and create new geometry
                edges_connecting = G.edges(node)
                assert len(edges_connecting) == edge_nr_to_consider, "Error: {} {} {} ".format(len(edges_connecting), edge_nr_to_consider, node)
                incoming_edge = list(edges_connecting)[0]
                outcoming_edge = list(edges_connecting)[1]

            if outcoming_edge == incoming_edge: # if same, then remove
                continue
            else:
                # Geometry incoming and outcoming edge (may already be multiple)
                geom_incoming = G.edges[incoming_edge]['geometry']
                geom_outcoming = G.edges[outcoming_edge]['geometry']

                merged_line = linemerge(
                    MultiLineString([LineString(i) for i in (geom_incoming, geom_outcoming)]))

                assert merged_line.type == 'LineString'

                # Check if new simplified is alreday in graph (then a loop). If yes, skip
                if (merged_line.coords[0], merged_line.coords[-1]) in G.edges():
                    continue
                else:

                    # Get attributes of both edges
                    incoming_attr = G.edges[incoming_edge]
                    outcoming_attr = G.edges[outcoming_edge]

                    # Iterate and copy if an attribute is in one but not the other
                    for attr in incoming_attr.keys():
                        if incoming_attr[attr] == None and outcoming_attr[attr] != None:
                            incoming_attr[attr] = outcoming_attr[attr]  # Overwrite
                        elif incoming_attr[attr] == 0 and outcoming_attr[attr] != 0: # if e.g. no bride and other is yes
                            incoming_attr[attr] = outcoming_attr[attr]  # Overwrite

                    incoming_attr['geometry'] = merged_line
                    nodes_to_remove.append(node)
                    
                    # --- Update graph
                    G.remove_edge(incoming_edge[0], incoming_edge[1])
                    G.remove_edge(outcoming_edge[0], outcoming_edge[1])
                    G.add_edge(merged_line.coords[0], merged_line.coords[-1])
                
                    for i, j in incoming_attr.items():
                        G.edges[(merged_line.coords[0], merged_line.coords[-1])][i] = j
        bar.next()
    bar.finish()

    # Remove old intermediary nodes
    for node in nodes_to_remove:
        G.remove_node(node)

    G = nx.compose(G, G_new_edges)

    # Add undirected again
    if if_directed_input:
        G = undirected_to_directed(G, label='tags.one')
    
    G.graph['crs'] = crs_from

    print("non geometric simplification finished", G)

    return G


def calc_edge_degree(G, label='edge_degree'):
    
    print("Calculating average density per edge")
    for edge in G.edges:
        start_node = edge[0]
        end_node = edge[1]
        density_start_node = G.nodes[start_node]['degree']
        density_end_node = G.nodes[end_node]['degree']
        G.edges[edge][label] = (density_start_node + density_end_node) / 2
    
    return G


def calc_GFA_density(
        G,
        buildings,
        radius=200,
        label='GFA_den',
        ha_factor=10000
    ):
    """Calculate percentage of ground floor area in circle
    1 --> 100% building cover
    0 --> 0% building cover
    """
    print(buildings.crs.srs)
    print(G.graph['crs'])
    assert buildings.crs.srs == G.graph['crs']

    print("Calculating search tree for buildings")
    rTree = index.Index()
    for edge_nr, centroid in enumerate(buildings.geometry.centroid):
        rTree.insert(edge_nr, centroid.bounds)

    tot_circle_area = radius * radius * math.pi / ha_factor  # Convert m2 to ha

    bar = Bar('Calculating GFA density:', max=G.number_of_nodes())
    nx.set_edge_attributes(G, 0, label)

    for node in G.nodes:
        circle = Point(node).buffer(radius)
        buildings_intersection = list(rTree.intersection(circle.bounds))
        building_area = 0
        intersection_area = 0

        for building_iloc in buildings_intersection:
            building_geometry = buildings.iloc[building_iloc].geometry

            # Only consider building with > 50% overlap of intersection
            if building_geometry.intersects(circle):
                intersection_area = building_geometry.intersection(circle)
                #p_overlap = (intersection_area / building_geometry.area)
                #if p_overlap > 0.5:
                building_area += intersection_area.area

        # Assign pop density
        building_area_ha =  building_area / ha_factor
        G.nodes[node][label] = building_area_ha / tot_circle_area

        bar.next()
    bar.finish()

    # Assign average population density per edge
    print("Calculating average density per edge")
    for edge in G.edges:
        density_start_node = G.nodes[edge[0]][label]
        density_end_node = G.nodes[edge[1]][label]
        G.edges[edge][label] = (density_start_node + density_end_node) / 2

    return G


def collect_average_edge_attribute(G, edges_path, attribute, norm=True):
    """Calculate average (normed with lengh or not) of edges
    """
    tot_l = 0
    sum_attributes_normed = 0
    sum_attributes = 0

    for edge in edges_path:
        length = G.edges[edge]['geometry'].length
        val = G.edges[edge][attribute]
        sum_attributes += G.edges[edge][attribute]
        sum_attributes_normed += length * val
        tot_l += length
    
    if norm:
        weighted_attribute = round(sum_attributes_normed / tot_l, 5)
    else:
        weighted_attribute = round(sum_attributes / len(edges_path), 5)

    return weighted_attribute


def calc_edge_and_node_pop_density(
        G,
        pop_points,
        radius,
        attribute_pop,
        label,
        ha_factor=10000
    ):
    """Buffer edge and calculate pop density
    
    Note: Works only with metric coordinate system

    Note: Returns population density per hectar
    """
    assert pop_points.crs.srs == G.graph['crs']

    tot_circle_area = radius * radius * math.pi / ha_factor  # Convert m2 to ha
        
    # Create search tree of pop_points
    print("Calculating search tree for pop_points")
    rTree = index.Index()
    for edge_nr, centroid in enumerate(pop_points.geometry.centroid):
        rTree.insert(edge_nr, centroid.bounds)

    nx.set_edge_attributes(G, 0, label)
    nx.set_node_attributes(G, 0, label)
 
    bar = Bar('Calculating population density:', max=G.number_of_edges())
    for edge in G.edges:
        edge_buffer = G.edges[edge]['geometry'].buffer(radius)
        edge_area = edge_buffer.area / ha_factor  # Convert m2 to ha
        pop_points_intersection = list(rTree.intersection(edge_buffer.bounds))

        tot_pop = 0
        for building_iloc in pop_points_intersection:
            pop_pnt = pop_points.iloc[building_iloc]
            if pop_pnt.geometry != None and pop_pnt.geometry.within(edge_buffer):
                tot_pop += int(pop_pnt[attribute_pop])

        # Assign pop density to edge
        G.edges[edge][label] = round(tot_pop / edge_area, 4)
        
        
        # Calcuate population denstity for node
        for node in edge:
            node_buffer = Point(node).buffer(radius)
            pop_points_intersection = list(rTree.intersection(node_buffer.bounds))
            tot_pop = 0
            for building_iloc in pop_points_intersection:
                pnt_fb = pop_points.iloc[building_iloc]
                if pnt_fb[attribute_pop] != None and pnt_fb.geometry.within(node_buffer):
                    tot_pop += int(pnt_fb[attribute_pop])
            
            # Assign pop density
            G.nodes[node][label] = round(tot_pop / tot_circle_area, 5)
            
        bar.next()
    bar.finish()

    return G


def calc_node_pop_density(
        G,
        buildings,
        radius,
        attribute_pop,
        label,
        ha_factor=10000
    ):
    """Iterate over node and assign number of population
    in a search radius (circle)
    
    Note: Works only with metric coordinate system

    Note: Returns population density per hectar
    """
    assert buildings.crs.srs == G.graph['crs']

    # Create search tree of buildings
    print("Calculating search tree for buildings")
    rTree = index.Index()
    for edge_nr, centroid in enumerate(buildings.geometry.centroid):
        rTree.insert(edge_nr, centroid.bounds)

    tot_circle_area = radius * radius * math.pi / ha_factor  # Convert m2 to ha

    nx.set_node_attributes(G, 0, label)

    bar = Bar('Calculating population density:', max=G.number_of_nodes())
    for node in G.nodes:
        node_buffer = Point(node).buffer(radius)
        buildings_intersection = list(rTree.intersection(node_buffer.bounds))
        tot_pop = 0
        for building_iloc in buildings_intersection:
            pnt_fb = buildings.iloc[building_iloc]
            if pnt_fb[attribute_pop] != None and pnt_fb.geometry.within(node_buffer):
                tot_pop += int(pnt_fb[attribute_pop])
        
        # Assign pop density
        G.nodes[node][label] = round(tot_pop / tot_circle_area, 5)

        bar.next()
    bar.finish()

    return G


def assign_attribute_by_largest_intersection(
        G_to_assign,
        G_base,
        min_intersection_d,
        crit_buffer,
        labels
    ):
    """Transfer attribute based on buffer
    """
    print("... create tre")
    rTree = build_rTree_G(G_base)
    all_edges = list(G_base.edges)

    bar = Bar('Adding attribute from one graph to parallel:', max=G_to_assign.number_of_edges())
    for edge in G_to_assign.edges:
        edge_to_check = G_to_assign.edges[edge]['geometry']
        edge_to_check_buffered = edge_to_check.buffer(crit_buffer)
        
        list_intersection_locs = list(rTree.intersection(edge_to_check_buffered.bounds))
        max_intersection_l = 0
        
        # Get list with longest intersection
        for loc in list_intersection_locs:
            intersection_edge = all_edges[loc]
            geom_edge = G_base.edges[intersection_edge]['geometry']
            if geom_edge.intersects(edge_to_check_buffered):
                intersection_l = edge_to_check_buffered.intersection(geom_edge).length
                
                if intersection_l > max_intersection_l:
                    edge_to_copy_attribute = intersection_edge
                    max_intersection_l = intersection_l

        # Copy attributes 
        if max_intersection_l > 0 and max_intersection_l > min_intersection_d:
            for label in labels:
                G_to_assign.edges[edge][label] = G_base.edges[edge_to_copy_attribute][label]

        bar.next()
    bar.finish()

    return G_to_assign


def check_if_paralell_lines(
        G_to_assign,
        G_base,
        crit_buffer,
        min_edge_distance,
        p_min_intersection,
        label
    ):
    """Iterate G_to_assign with coordinate tuple nodes.

    crit_buffer: [m] Buffer distance to test spatial overlay 
    Buffer the edges and check whether a network edge is in the buffered edge

    Note: The G_to_assign should have individual segements
    (LineString rather than MultiLineString)
    """
    print("...Merge all geometries of tram/bus/geomtries")
    geometries = []
    for edge in G_to_assign.edges:
        geometries.append(G_to_assign.edges[edge]['geometry'])
    joined_geometry = unary_union(geometries)
    joined_geometry_buffer = joined_geometry.buffer(crit_buffer)
    
    bar = Bar('Adding attribute from one graph to parallel:', max=G_base.number_of_edges())
    for edge in G_base.edges:
        edge_to_check = G_base.edges[edge]['geometry']

        # Total line length
        tot_length = edge_to_check.length

        if joined_geometry_buffer.contains(edge_to_check):
            # Only assign lavel if has a certain length
            if edge_to_check.length >= min_edge_distance:
                G_base.edges[edge][label] = 1
            else:
                G_base.edges[edge][label] = 0
        elif joined_geometry_buffer.intersects(edge_to_check):
            intersection_length = joined_geometry_buffer.intersection(edge_to_check).length

            # percentage intersected
            p_intesect = (100 / tot_length) * intersection_length

            if p_intesect >= p_min_intersection:
                if edge_to_check.length >= min_edge_distance:
                    G_base.edges[edge][label] = 1
                else:
                    G_base.edges[edge][label] = 0

        bar.next()
    bar.finish()

    return G_base


def gdf_multipolygon_to_polygon(gdf, only_geom=False):
    """
    """
    assert gdf.index.is_unique

    attributes = gdf.columns.tolist()
    geometry_loc_attr = attributes.index('geometry')
    attribute_list = []

    for index_gdf in gdf.index:
        geometry = gdf.loc[index_gdf].geometry
        attributes_original = gdf.loc[index_gdf].values
        new_attributes = []
        if geometry.type == 'MultiPolygon':
            for geometry in geometry:
                new_attribute = copy.copy(attributes_original)
                new_attribute[geometry_loc_attr] = geometry
                attribute_list.append(new_attribute)
                #new_attributes.append(geometry)
                #attribute_list.append(geometry)
        else:
            attribute_list.append(attributes_original)

    gdf_multipolyon = gpd.GeoDataFrame(attribute_list, columns=attributes, crs=gdf.crs)

    if only_geom:
        gdf_multipolyon = gdf_multipolyon[['geometry']]

    return gdf_multipolyon

def unsimplify(G, G_simplified_original):
    """If a network has been simpilfied, check
    which edges and and compare with other graph.
    Add missing nodes and split the originally simplified
    edges
    """
    simplified_edges_to_remove = []
    G_intermediary = nx.Graph()
    G_intermediary.graph = G.graph
    original_nodes = list(G_simplified_original.nodes)

    # Iterated edges
    for edge in G.edges:
        if edge in G_simplified_original.edges:
            pass # no problem, exactly same edge geometry
        else:  # edge was simplified
            edge_coors = list(G.edges[edge]['geometry'].coords)
            edge_attributes = G.edges[edge]

            # Remove edge evenutally
            simplified_edges_to_remove.append(edge)
            
            # Get all nodes on edge and check which one was simplified
            nodes_on_edge = []
            for node in edge_coors:
                if node in original_nodes:
                    nodes_on_edge.append(node)
                    G_intermediary.add_node(node)

            #assert len(nodes_on_edge) == len(set(nodes_on_edge))
            # Get all edges of pnts on edge
            for node in nodes_on_edge:
                neigbor_edges = list(G_simplified_original.edges(node))
                for neigbor_edge in neigbor_edges:
                    if neigbor_edge not in G_intermediary.edges:
                        
                        # Get only neighbour wthih are on line
                        if neigbor_edge[0] in nodes_on_edge and neigbor_edge[1] in nodes_on_edge:
                            G_intermediary.add_edge(neigbor_edge[0], neigbor_edge[1])

                            # Copy geometry attributes of simplified to original
                            G_intermediary.edges[neigbor_edge]['geometry'] = G_simplified_original.edges[neigbor_edge]['geometry']

                            for attribute, value in edge_attributes.items():
                                if attribute != 'geometry' and attribute != 'wkt' and attribute != 'Json':
                                    G_intermediary.edges[neigbor_edge][attribute] = value

    '''nodes, edges = hp_rw.nx_to_gdf(G)
    edges.to_file("C:/_scrap/X11.shp")
    nodes, edges = hp_rw.nx_to_gdf(G_intermediary)
    edges.to_file("C:/_scrap/XG_intermediary.shp")'''

    # Remove edges
    for edge in simplified_edges_to_remove:
        G.remove_edge(edge[0], edge[1])

    G = nx.compose(G, G_intermediary)

    return G


def G_multilinestring_to_linestring(G, only_geom=False, single_segments=True):
    """MulitlineSTring to lines --> indiviual segments of lines

    single_segments : convert all lines with multiple coordinates (even though SingleLine)
    Note: All non-geometry attributes get assigned copied for each new segment
    """
    crs_orig = G.graph['crs']
    edges_to_remove = []

    new_Graph = nx.Graph()
    new_edges = []
    for edge_index in G.edges:
        geometry = G.edges[edge_index]['geometry']
        attributes_original = G.edges[edge_index]

        if (geometry.type == 'MultiLineString'): # or (nr_of_segments > 2 and single_segments):
            #line_segments = helper_MultiLineString_to_LineString(x=geometry.coords.xy[0].tolist(), y=geometry.coords.xy[1].tolist())

            for line in geometry:
                new_attribute = copy.copy(attributes_original)
                new_attribute['geometry'] = line
                new_Graph.add_node(line.coords[0])
                new_Graph.add_node(line.coords[1])
                new_Graph.add_edge(line.coords[0], line.coords[1])
                for attr, val in new_attribute.items():
                    new_Graph.edges[(line.coords[0], line.coords[1])][attr] = val

            #G.remove_edge(edge_index[0], edge_index[1])
            edges_to_remove.append((edge_index[0], edge_index[1]))

    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])
            
    G = nx.compose(G, new_Graph)       
         
    return G


def gdf_multilinestring_to_linestring(gdf, only_geom=False, single_segments=True):
    """MulitlineSTring to lines --> indiviual segments of lines

    single_segments : convert all lines with multiple coordinates (even though SingleLine)
    Note: All non-geometry attributes get assigned copied for each new segment
    """
    crs_orig = gdf.crs

    assert gdf.index.is_unique

    attributes = gdf.columns.tolist()
    geometry_loc_attr = attributes.index('geometry')
    attribute_list = []

    for index_gdf in gdf.index:
        geometry = gdf.loc[index_gdf].geometry
        attributes_original = gdf.loc[index_gdf].values

        if (geometry.type == 'MultiLineString'):
            for line in geometry:
                new_attribute = copy.copy(attributes_original)
                new_attribute[geometry_loc_attr] = line
                attribute_list.append(new_attribute)
        else:
            attribute_list.append(attributes_original)

    gdf_linestring = gpd.GeoDataFrame(attribute_list, columns=attributes, crs=crs_orig)

    if only_geom:
        gdf_linestring = gdf_linestring[['geometry']]

    return gdf_linestring

def build_rTree_G(G):
    """Buid rTree
    """
    rTree = index.Index()
    for cnt, edge in enumerate(G.edges):
        rTree.insert(cnt, G.edges[edge]['geometry'].bounds)

    return rTree


def build_rTree(df, index_type='iloc'):
    """Buid rTree

    iloc: rTree gets built with ilocation
    loc: rTree gets built with index
    """
    rTree = index.Index()

    if index_type == 'iloc':
        for pos, poly in enumerate(df.geometry):
            rTree.insert(pos, poly.bounds)
    elif index_type == 'loc':
        for pos in df.index:
            poly = df.loc[pos].geometry
            rTree.insert(pos, poly.bounds)
    else:
        raise Exception("Wrong index_type defined")
    return rTree


def remove_all_intersections(gdf):
    """
    1. unary_union first
    2. Iterate nodes and copy dominant attributes which intersects line
    """
    # Attributes
    gdf_attributes = gdf.columns.tolist()

    # Create tree of old edges
    gdf_tree = build_rTree(gdf, index_type='iloc')

    # Merge all lines
    removed_intersections = unary_union(gdf['geometry'].tolist())
    gdf_without_columns = gpd.GeoDataFrame(removed_intersections, columns=['geometry'])
    gdf_without_columns = gdf_without_columns.reset_index(drop=True)

    attributes_list = []
    bar = Bar('Checking if intersections: ', max=gdf_without_columns.shape[0])

    for gdf_index in gdf_without_columns.index:
        line_to_check = gdf_without_columns.loc[gdf_index].geometry
        lines_bb = list(gdf_tree.intersection(line_to_check.bounds))
        #lines_bb = list(gdf_tree.contains(line_to_check.bounds))

        crit_overlap = False
        # Get line with largest intersection
        intersection_length = 0
        for tree_index in lines_bb:
            line_in_bb = gdf.iloc[tree_index].geometry

            #if line_to_check.buffer(0.5).contains(line_in_bb):
            if line_in_bb.buffer(0.5).contains(line_to_check):
                attributes = gdf.iloc[tree_index].values.tolist()
                crit_overlap = True
                break
            '''else:
                # Intersection but no crossing
                # Note use small buffer to make sure that intersection
                intersection_obj = line_to_check.buffer(0.5).intersection(line_in_bb)
                length = intersection_obj.length
                if gdf.iloc[tree_index]['tags.name'] == 'TESTSTR':
                    print("INTERLEN: {}".format(length))
                    print("-")

                if intersection_obj.type == 'LineString' and length > intersection_length:
                    attributes = gdf.iloc[tree_index].values.tolist()
                    crit_overlap = True
                    intersection_length = length
                else:
                    pass
            '''
        if crit_overlap:
            attributes_list.append(attributes)
        else:
            raise Exception("No matching line found for trasnferring attributes")
            print("No matching line found for trasnferring attributes")
        bar.next()
    bar.finish()

    df_attributes = pd.DataFrame(attributes_list, columns=gdf_attributes)  
    df_attributes = df_attributes.drop(columns='geometry')
    
    gpd_no_intersections = gpd.GeoDataFrame(
        df_attributes, geometry=gdf_without_columns.geometry, crs=gdf.crs)
    
    return gpd_no_intersections     


def clean_up_intersections(gdf):
    """Add nodes to all intersections

    Note: Removes onle simple intersections (not if multiple edges cut same line)
    """
    gdf.reset_index(drop=True) # reset index

    assert gdf.index.is_unique

    # Create search tree with network
    gdf_tree = build_rTree(gdf, index_type='iloc')

    bar = Bar('Checking if intersections: ', max=gdf.shape[0])

    gdf_columns = gdf.columns.tolist()
    geometry_pos = gdf_columns.index('geometry')

    new_lines = []
    cnt_intersections = 0
    index_to_remove = []
    for gdf_index in gdf.index:

        # Get all intersecting lines with bb
        line_to_check = gdf.loc[gdf_index].geometry
        lines_bb = gdf_tree.intersection(line_to_check.bounds)

        # Iterate lines in bb and check whether intersect
        for tree_index in lines_bb:
            line_in_bb = gdf.iloc[tree_index].geometry
            if line_to_check.crosses(line_in_bb):
                intersection_pnts = line_to_check.intersection(line_in_bb)
                cnt_intersections += 1
                new_line_1 = gdf.iloc[tree_index].values.tolist()
                new_line_2 = gdf.loc[gdf_index].values.tolist()
                if intersection_pnts.geom_type == 'MultiPoint':
                    for intersection_pnt in intersection_pnts:
                        # Add lines from intersection point to start and end of each line
                        new_line_1[geometry_pos] = LineString(((line_to_check.coords[0][0], line_to_check.coords[0][1]), (intersection_pnt.x, intersection_pnt.y)))
                        new_lines.append(new_line_1)
                        new_line_1[geometry_pos] = LineString(((intersection_pnt.x, intersection_pnt.y), (line_to_check.coords[-1][0], line_to_check.coords[-1][1])))
                        new_lines.append(new_line_1)
                        new_line_2[geometry_pos] = LineString(((line_in_bb.coords[0][0], line_in_bb.coords[0][1]), (intersection_pnt.x, intersection_pnt.y)))
                        new_lines.append(new_line_2)
                        new_line_2[geometry_pos] = LineString(((intersection_pnt.x, intersection_pnt.y), (line_in_bb.coords[-1][0], line_in_bb.coords[-1][1])))
                        new_lines.append(new_line_2)
                        #new_geometies.append([LineString(((line_to_check.coords[0][0], line_to_check.coords[0][1]), (intersection_pnt.x, intersection_pnt.y)))])
                        #new_geometies.append([LineString(((intersection_pnt.x, intersection_pnt.y), (line_to_check.coords[-1][0], line_to_check.coords[-1][1])))])
                        #new_geometies.append([LineString(((line_in_bb.coords[0][0], line_in_bb.coords[0][1]), (intersection_pnt.x, intersection_pnt.y)))])
                        #new_geometies.append([LineString(((intersection_pnt.x, intersection_pnt.y), (line_in_bb.coords[-1][0], line_in_bb.coords[-1][1])))])
                else:
                    # Add lines from intersection point to start and end of each line
                    new_line_1[geometry_pos] = LineString(((line_to_check.coords[0][0], line_to_check.coords[0][1]), (intersection_pnts.x, intersection_pnts.y)))
                    new_lines.append(new_line_1)
                    new_line_1[geometry_pos] = LineString(((intersection_pnts.x, intersection_pnts.y), (line_to_check.coords[-1][0], line_to_check.coords[-1][1])))
                    new_lines.append(new_line_1)
                    new_line_2[geometry_pos] = LineString(((line_in_bb.coords[0][0], line_in_bb.coords[0][1]), (intersection_pnts.x, intersection_pnts.y)))
                    new_lines.append(new_line_2)
                    new_line_2[geometry_pos] = LineString(((intersection_pnts.x, intersection_pnts.y), (line_in_bb.coords[-1][0], line_in_bb.coords[-1][1])))
                    new_lines.append(new_line_2)
                    #new_geometies.append([LineString(((line_to_check.coords[0][0], line_to_check.coords[0][1]), (intersection_pnts.x, intersection_pnts.y)))])
                    #new_geometies.append([LineString(((intersection_pnts.x, intersection_pnts.y), (line_to_check.coords[-1][0], line_to_check.coords[-1][1])))])
                    #new_geometies.append([LineString(((line_in_bb.coords[0][0], line_in_bb.coords[0][1]), (intersection_pnts.x, intersection_pnts.y)))])
                    #new_geometies.append([LineString(((intersection_pnts.x, intersection_pnts.y), (line_in_bb.coords[-1][0], line_in_bb.coords[-1][1])))])

                index_to_remove.append(gdf_index)
                index_to_remove.append(tree_index)

        bar.next()
    bar.finish()

    # Remove cutted lines
    gdf = gdf.drop(index=index_to_remove)

    # Add new lines
    #new_geomtries_gdf = gpd.GeoDataFrame(new_geometies, columns=['geometry'])
    new_geomtries_gdf = gpd.GeoDataFrame(new_lines, columns=gdf_columns)

    gdf = gdf.append(new_geomtries_gdf)
    gdf = gdf.reset_index(drop=True)
    print("Number of intersection incidents: {}".format(cnt_intersections))
    assert gdf.index.is_unique

    return gdf


def clean_merge_close(G, from_node, to_node): #merge_distance=10):
    """Add all path from from_node to to_node
    and update network
    Improve: Update in middle of geometries
    """
    # Get all path of from_node
    neighbors_from_node = G.neighbors(from_node)

    # Remove existing edges
    for neighbour in neighbors_from_node:
        try:
            G.remove_edge((from_node, neighbour))
        except:
            pass
        try:
            G.remove_edge((neighbour, from_node))
        except:
            pass

    # Add new edges
    for neighbour in neighbors_from_node:
        G.add_edge((neighbour, to_node))
        G.add_edge((to_node, neighbour))

    # Remove
    G.remove_node(from_node)

    return G


def gdf_add_edge_geometry_from_key(edges):
    """From a gdf without geometry but use stored
    geometric information and add as geometry attribute

    Note: must be used as momepy.nx_to_gdf does not write out geometry
    """
    edge_geometries = []
    for index in edges.index:
        geometry_edge = LineString(edges.loc[index]['coord_from_key'])
        #geometry_edge = LineString(edges.loc[index]['node_start'], edges.loc[index]['node_end'],)
        if not geometry_edge.is_valid: # Test if valid geometry
            raise Exception("Not valid edge: {}".format(geometry_edge))
        edge_geometries.append(geometry_edge)
    edges = edges.drop(columns=['coord_from_key'])
    edges['geometry'] = edge_geometries

    return edges

def edges_to_nodes(list):
    """
    [(1,2), (2,3), (3, 4)] --> [1,2,3,4]
    """
    node_list = [list[0][0]]

    for edge in enumerate(list):
        node_list.append(edge[1])

    return node_list


def to_tuple_list(list):
    """
    [1,2,3,4] --> [(1,2), (2,3), (3, 4)]
    """
    tuple_list = []

    for cnt, i in enumerate(list):
        if cnt > 0:
            tuple_list.append((prev, i))
        prev = i

    return tuple_list


def clean_osm_tags(roads):
    """Clean OSM tags
    """
    tags_to_add = {
        "oneway": 0} # tag: default_value
    assert roads.index.is_unique

    for tag_to_add, default in tags_to_add.items():
        roads[tag_to_add] = default  # add default

        # Iterate and add tag if available
        for index in roads.index:
            tag_full_string = roads.loc[index]['other_tags']
            if tag_full_string:
                tag_str = tag_full_string.split(",")

                for tag in tag_str:
                    if tag_to_add in tag:
                        value = tag.split("=>")[1].replace('"', '').strip()
                        if value == 'yes':
                            value = 1
                        elif value == 'no':
                            value = 0
                        else:
                            value = value

                        roads.loc[index, tag_to_add] = value

    return roads


def add_tuple_key_as_xy(G):
    """Add coordinates as attributes to node
    
    Note: Start with enumeration at 1 in order to 
    be identical with nx_to_gdf method
    
    """
    start_enumeration = 1
    dict_coord = {}
    for node_id, node in enumerate(G.nodes, start_enumeration):
        G.nodes[node]['x'] = node[0]
        G.nodes[node]['y'] = node[1]
        dict_coord[node_id] = node

    return G, dict_coord


def create_clean_graph(G_roads, graph_type='Graph'):
    """Take readings from networkx read_shp
    and generate a MultiDiGraph with coordinates
    and node numbers

    Example
    -------
    node 1 : {x: coordinate, y: coordinate}
    """
    # Get node position coordinates
    shp_pos_coor = {}
    shp_pos_coor_inverse = {}
    for iter, coordinates in enumerate(G_roads.nodes()):
        shp_pos_coor[iter] = coordinates
        shp_pos_coor_inverse[coordinates] = iter  # inverse storage

    # Create empty Graph
    if graph_type == 'Graph':
        X = nx.Graph()
    elif graph_type == 'DiGraph':
        X = nx.DiGraph()  # MultiDiGraph
    else:
        raise Exception("Wrong type defined")   

    #Add nodes preserving coordinates
    X.add_nodes_from(shp_pos_coor.keys())

    # Add coordinates as attributes to node
    for node_nr in X.nodes:
        X.nodes[node_nr]['x'] = shp_pos_coor[node_nr][0]
        X.nodes[node_nr]['y'] = shp_pos_coor[node_nr][1]

    #l = [set(x) for x in G_roads.edges()] #To speed things up in case of large object

    # Map the G.edges start and endpoints onto shp_pos_coor
    edg = []
    for edge_coordinates in G_roads.edges():
        from_remap = shp_pos_coor_inverse[edge_coordinates[0]]
        to_remap = shp_pos_coor_inverse[edge_coordinates[1]]
        edg.append((from_remap, to_remap))

        # if DiGraph
        #edg.append((to_remap, from_remap))
    X.add_edges_from(edg)

    return X, shp_pos_coor


def project(gdf, to_crs):
    """Project
    
    Info: Different CRS handling in newer geopanda version
    https://geopandas.org/projections.html
    """
    from_crs = gdf.crs.srs

    if from_crs != 'epsg:{}'.format(to_crs):
        gdf = gdf.to_crs("epsg:{}".format(to_crs))

    return gdf

def test_if_only_lines(gdf):
    """Test if point
    """
    for type in gdf.geometry.type:
        if type != 'LineString':
            raise Exception("Point in line gdf")


def drop_z_linestring(linedata):
    """LINESTRING Z to LINESTRING
    """

    bar_main = Bar('... dropping z values', max=linedata.shape[0])
    new_geometries = []
    for i in linedata.index:
        line_3d = linedata.loc[i].geometry
        line_2d = shapely.wkb.loads(shapely.wkb.dumps(line_3d, output_dimension=2))
        new_geometries.append(line_2d)
        bar_main.next()
    bar_main.finish()

    linedata['geometry'] = new_geometries
    return linedata


def remove_holes(input_gdf):
    """Remove all holes in polygons (interiors) 
    """
    for index_nr in input_gdf.index:
        poly = input_gdf.loc[index_nr].geometry

        if len(list(poly.interiors)) > 0:

            internal_rings_to_keep = []
            for internal_ring in (poly.interiors):
                area_internal_ring = Polygon(internal_ring).area

            if len(internal_rings_to_keep) > 0:
                new_polygon = Polygon(poly.exterior, internal_rings_to_keep)
                input_gdf.at[index_nr, 'geometry'] = new_polygon
            else:
                # Replace geometry with exterior
                input_gdf.at[index_nr, 'geometry'] = Polygon(poly.exterior)

    input_gdf = input_gdf.reset_index(drop=True)

    return input_gdf

