"""
Main file. Before running this file, openstreetmap data needs to be downloaded by preprocess_osm

"""
import os
import sys

import warnings
warnings.filterwarnings("ignore")

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_superblocks)
from rtree import index
import logging
import networkx as nx
import geopandas as gpd
import yaml

from sqlalchemy import create_engine

to_crs_meter = 32632  # 2056: CH1903+ / LV95   4326: WSG 84

from superblocks.scripts.network import helper_network as hp_net
from superblocks.scripts.network import helper_osm as hp_osm
from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import helper_scenario as hp_sc
from superblocks.scripts.network import flow_independent as flow_indep

# Parallelization
parallel_mode = False
local = False
swiss_example = False

# Initial steps
cleaning_and_simplifying = True
centrality_calculations = True
calculation_of_indicators = True

# Classification
network_step_analysis_blocktype = True
final_classification = True

# Write shapefiles of blocks
execute_flow = True
write_interventions = True

# Config
write_anyway = False
crit_bus_is_big_street = False
radius_pop_density = 100
radius_GFA_density = 100
big_road_labels = ['primary_link', 'primary', 'secondary_link', 'secondary', 'trunk', 'trunk_link'] #, 'tertiary', 'tertiary_link']
other_streets = ['pedestrian', 'living_street', 'service']

# Writing out of blocks
mode_write_individual = True   # True: Create unique shapes for each indivdiual block
crit_only_full = False          # True: Only full interventions (no overlap)   False: Allow overlaps (but not fully contained)
type_cadastre = True

# Label names
tag_id_superb = 'id_superb'
tag_id_minib = 'id_mini'
tag_id_miniS = 'id_miniS'
label_superb = 'b_superb'
label_mini = 'b_mini'
label_miniS = 'b_miniS'

# [decimal] Percentage how much deviation form origianl barcelnoian superblock
crit_deviations = [1.0]

# Define for which crit_deviation the superblocks are written out
write_out_scenario = 1.0

# Writing out block options
crit_write_street_area = True
crit_write_buildings = True
crit_write_street_negative = True

path_city_raw = "/data/cities"
path_results = "/data/results/results_superblock"

cities = [
    #'atlanta', 
    #'bankok',
    #'barcelona',
    'berlin',
    #'budapest',
    #'cairo',
    #'hong_kong',
    #'lagos',
    #'london',
    #'madrid',
    #'melbourne',
    #'mexico_city',
    #'paris',
    #'rome',
    #'sydney', 
    #'tokyo',
    #'warsaw',
    #'zurich',
    #'frankfurt',
    #'freiburg',
    #'hamburg'
    #'munchen'
    ]

initial_street_name = "street_network_edges_with_attributes_pop_density.shp"

if parallel_mode:
    counter = int(sys.argv[1])
    print("Counter info: {}".format(counter))

    # Counter list including deviations
    counter_list = []
    for crit in crit_deviations:
        for city in cities:
            counter_list.append((city, crit))

    cities = [counter_list[counter][0]]             # City
    crit_deviations = [counter_list[counter][1]]    # Crit deviation

print("------")
print(cities)
print(crit_deviations)
print("------")

postgis_connection = create_engine(f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
                                   f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}" \
                                   f"@{os.getenv('POSTGRES_HOST', 'localhost')}:5432/{os.getenv('POSTGRES_DATABASE', 'postgres')}")


for city in cities:
    print("Case study: {}".format(city))
    path_in = os.path.join(path_city_raw, str(city))
    path_temp = os.path.join(path_results, str(city))
    path_temp_classified = os.path.join(path_results, str(city), 'streets_classified')
    path_blocks = os.path.join(path_temp, 'blocks')
    hp_rw.create_folder(path_blocks)   
    hp_rw.create_folder(path_temp)
    hp_rw.create_folder(path_temp_classified)

    hp_rw.set_up_logger(os.path.join(path_temp, 'logger.log'))
    logger = logging.getLogger('Fiona')
    logger.setLevel("WARNING")

    # ==============================================================================
    # (I) Cleaning and simplifying graph geometrically simplify street network
    # =============================================================================
    if cleaning_and_simplifying:

        # Parameters
        max_length_driveway = 15            # [m]
        max_distance_node_merging = 15      # [m]

        # ==============================================================================
        # Create Street Graph
        # ==============================================================================
        # TODO:
        #   (1) get BBox
        #   (2) get intersecting roads, buildings,
        #gdf_roads = gpd.read_file(os.path.join(path_in, initial_street_name))

        # TODO: add where clause with bounding box
        gdf_roads = gpd.read_postgis("SELECT * FROM street_network_edges_with_attributes_pop_density", postgis_connection, geom_col="geometry")

        # Remove Z attribute if it has z geometry
        if list(gdf_roads.has_z)[0]:
            gdf_roads = hp_net.flatten_geometry(gdf_roads)

        # Make selection of attributes
        attributes_to_keep_tested = []
        attributes_to_keep = [
            'geometry', 'type', 'tags.access', 'tags.highway', 'tags.tunnel', 'tags.maxspeed',
            'tags.name', 'tram', 'bus', 'trolleybus', 'GFA_den', 'pop_den', 'tags.bridge',
            'DWV_FZG', 'DWG_PW']

        for attribute in attributes_to_keep:
            if attribute in gdf_roads.columns.tolist():
                attributes_to_keep_tested.append(attribute)

        gdf_roads = gdf_roads[attributes_to_keep_tested]
        gdf_roads.to_file("/data/tmp/_scrap/first.shp")
        G_roads = hp_rw.gdf_to_nx(gdf_roads)

        # ----Add bridges attribute, distance
        tags_to_remodel = ['tags.bridge', 'tags.tunnel']
        for tag_to_remodel in tags_to_remodel:
            if tag_to_remodel in gdf_roads.columns.tolist():
                for edge in G_roads.edges:
                    if G_roads.edges[edge][tag_to_remodel] == 'yes' or G_roads.edges[edge][tag_to_remodel] == 1:
                        G_roads.edges[edge][tag_to_remodel] = 1
                    else:
                        G_roads.edges[edge][tag_to_remodel] = 0
            else:
                gdf_bridges = gpd.read_postgis("SELECT * FROM bridges", postgis_connection, geom_col="geometry") # TODO: add where clause + bbox
                if gdf_bridges.shape[0] > 0:
                    #gdf_bridges = gpd.read_file(os.path.join(path_in, "bridges.shp"))
                    G_roads = hp_net.add_attribute_intersection(G_roads, gdf_bridges, label='tags.man_made', label_new=tag_to_remodel)
                    for edge in G_roads.edges:
                        if G_roads.edges[edge][tag_to_remodel] == 'bridge':
                            G_roads.edges[edge][tag_to_remodel] = 1
                else:
                    nx.set_edge_attributes(G_roads, None, tag_to_remodel)

        # --- Convert DiGraph to Graph
        G_roads_frozen = nx.to_undirected(G_roads)
        G_roads = nx.Graph(G_roads_frozen)

        # Relabel road if service but has tram on it --> unclassified
        for edge in G_roads.edges:
            if G_roads.edges[edge]['tags.highway'] == 'service' and G_roads.edges[edge]['tram'] == 1:
                G_roads.edges[edge]['tags.highway'] = 'unclassified'

        # Remove all private streets and footways
        G_roads = hp_net.remove_edge_by_attribute(G_roads, attribute='tags.highway', value="footway")
        G_roads = hp_net.remove_edge_by_attribute(G_roads, attribute='tags.access', value="private")
        G_roads = hp_net.remove_edge_by_attribute(G_roads, attribute='tags.highway', value="service")

        # Remove nodes which are on top of a building and degree 1
        buildings = gpd.read_postgis("SELECT * FROM buildings", postgis_connection, geom_col="geometry")
        #buildings = gpd.read_file(os.path.join(path_in, "osm_buildings.shp"))
        buildings = hp_osm.remove_faulty_polygons(buildings)
        G_roads = hp_net.remove_intersect_building(G_roads, buildings)

        # To single segments for cleaning steps
        G_roads = hp_net.G_multilinestring_to_linestring(G_roads, single_segments=True)
        nodes, edges = hp_rw.nx_to_gdf(G_roads)
        edges = hp_net.remove_rings(edges)

        # Simplify
        G_roads = hp_rw.gdf_to_nx(edges)
        G_roads = hp_net.simplify_network(
            G_roads, crit_bus_is_big_street=crit_bus_is_big_street)
        G_roads = hp_net.calculate_node_degree(G_roads)

        # Geometric simplification
        G_roads = hp_net.geometric_simplification(G_roads, max_distance=max_distance_node_merging)
        G_roads = hp_net.calculate_node_degree(G_roads)
        G_roads = hp_net.simplify_network(G_roads, crit_bus_is_big_street=crit_bus_is_big_street)

        # Remove self loops
        G_roads.remove_edges_from(nx.selfloop_edges(G_roads))
        G_roads = hp_net.calculate_node_degree(G_roads)

        # Remove small zufahrt
        G_roads = hp_net.remove_zufahrt(G_roads, max_length=max_length_driveway)
        G_roads = hp_net.simplify_network(G_roads, crit_bus_is_big_street=crit_bus_is_big_street)

        nodes, edges = hp_rw.nx_to_gdf(G_roads)
        #edges.to_file(os.path.join(path_temp, 'simplified_edges.shp'))
        #nodes.to_file(os.path.join(path_temp, 'simplified_nodes.shp'))

        edges.to_postgis("simplified_edges", postgis_connection, if_exists="replace")
        nodes.to_postgis("simplified_nodes", postgis_connection, if_exists="replace")
        print("...Simplified network")

    # ==============================================================================
    # Superblock classification
    # ==============================================================================
    if calculation_of_indicators:
        gdf_roads = edges
        #gdf_roads = gpd.read_file(os.path.join(path_temp, "simplified_edges.shp"))

        G = hp_rw.gdf_to_nx(gdf_roads)
        G = nx.to_undirected(G)
        G = nx.Graph(G)

        # Calculate measures
        G = hp_net.calc_eccentricity(G, 'eccentr')
        G = hp_net.calc_edge_connectivity(G, label='loc_conn')

        G = hp_net.find_cycles(G)
        G = hp_net.calculate_node_degree(G)
        G = hp_net.calculate_closeness_centrality(G)
        G = hp_net.calculate_average_edge_attribute_from_node(G, label='closeness_centrality')

        G_frozen = nx.to_undirected(G)
        G = nx.Graph(G_frozen)

        # Add new attributes
        nx.set_edge_attributes(G, 0, "superblocks")
        nx.set_edge_attributes(G, None, 'inter_typ')
        nx.set_edge_attributes(G, None, tag_id_miniS)
        nx.set_edge_attributes(G, None, tag_id_minib)
        nx.set_edge_attributes(G, None, tag_id_superb)

        # ==============================================================================
        # Prepare street network for superblock analysis
        # ==============================================================================
        hp_net.remove_edge_by_attribute(G, attribute='tags.highway', value="pedestrian")

        # Remove edges with local connectivity == 1 (individual cul-de-sac streets)
        # Local edge connectivity: Local edge connectivity for two nodes s and t is the minimum 
        # numberof edges that must be removed to disconnect them. Remove all edges witch connectivity == 1
        G = hp_net.remove_edge_by_attribute(G, 'loc_conn', 1)
        G.remove_nodes_from(list(nx.isolates(G)))

        G = hp_net.simplify_network(G, crit_bus_is_big_street=crit_bus_is_big_street)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = hp_net.calculate_node_degree(G)

        # Recalculate population density after simplifying network
        pop_pnts = gpd.read_postgis("SELECT * FROM fb_pop", postgis_connection, geom_col="geometry")
        G = hp_net.calc_edge_and_node_pop_density(G, pop_pnts, radius=radius_pop_density, attribute_pop='population', label='pop_den')

        # Calculate GFA density based on osm buildings
        buildings_osm = gpd.read_postgis("SELECT * FROM buildings", postgis_connection, geom_col="geometry")
        G = hp_net.calc_GFA_density(G, buildings_osm, radius=radius_GFA_density, label='GFA_den')

        nodes, edges = hp_rw.nx_to_gdf(G)
        #edges.to_file(os.path.join(path_temp, 'cleaned_edges.shp'))
        #nodes.to_file(os.path.join(path_temp, 'cleaned_nodes.shp'))

        edges.to_postgis("cleaned_edges", postgis_connection, if_exists="replace")
        nodes.to_postgis("cleaned_nodes", postgis_connection, if_exists="replace")

    # ---------------------------------------------------------
    # Classify edges
    # ---------------------------------------------------------
    if network_step_analysis_blocktype:
        #gdf_roads = gpd.read_file(os.path.join(path_temp, "cleaned_edges.shp"))
        #gdf_nodes = gpd.read_file(os.path.join(path_temp, "cleaned_nodes.shp"))
        gdf_roads = edges
        gdf_nodes = nodes
        G = hp_rw.gdf_to_nx(gdf_roads)

        # Append attributes
        for index_gdf in gdf_nodes.index:
            entry = gdf_nodes.loc[index_gdf]
            node = (entry.geometry.centroid.x, entry.geometry.centroid.y)
            G.nodes[node]['pop_den'] = entry['pop_den']
            G.nodes[node]['GFA_den'] = entry['GFA_den']

        # ------------------------------------------------------------------------------------------
        # Get subgraphs depending on node connectivity
        # Iterate node with > 4 edges and add if on cycle/loop (only undirected)
        # Note: G_deg3_4 also contains nodes with degree 3 (alls 4x4 nodes and thier influcing edges)
        # ------------------------------------------------------------------------------------------
        G_deg3_4, _ = hp_net.get_subgraph_degree(G, degree_nr=3, method='edge_neighbours')
        nodes_degree_3_4 = list(G_deg3_4.nodes)

        # ------------------------------------------------------------------------------------------
        # Remove all nodes which are on big streets, which are bridges, or which are tram/trolley lanes
        # ------------------------------------------------------------------------------------------
        nodes_to_ignore = []
        for ij, node4deg in enumerate(nodes_degree_3_4):
            neighbors = list(G.neighbors(node4deg))
            for neighbor in neighbors:
                edge = (node4deg, neighbor)
                highway = G.edges[edge]['tags.highway']
                crit_tram = G.edges[edge]['tram']
                if crit_bus_is_big_street:
                    crit_bus = G.edges[edge]['bus']
                else:
                    crit_bus = False
                crit_trolley = G.edges[edge]['trolleybus']
                crit_bridge = G.edges[edge]['tags.bridge']
                crit_tunnel = G.edges[edge]['tags.tunnel']
                if highway in big_road_labels or crit_tram or crit_trolley or crit_bus or crit_bridge == 1: #or crit_tunnel
                    nodes_to_ignore.append(node4deg)

        # Remove edges which are between nodes which should be ignored
        for node_to_ignore in nodes_to_ignore:
            neigbors_ignore = list(G_deg3_4.neighbors(node_to_ignore))
            for neighbor_ignore in neigbors_ignore:
                if neighbor_ignore in nodes_to_ignore:
                    edge_to_remove = (node_to_ignore, neighbor_ignore)
                    if edge_to_remove in G_deg3_4.edges:
                        G_deg3_4.remove_edge(node_to_ignore, edge_to_remove[1])

        # Remove isolated nodes
        G_deg3_4.remove_nodes_from(list(nx.isolates(G_deg3_4)))

        # After having removed the nodes_to_ignore, reselect network to build cycles
        G_deg3_4, _ = hp_net.get_subgraph_degree(G_deg3_4, degree_nr=3, method='edge_neighbours')

        # --- Calculate edge indicators on sub-network
        G_roads = hp_net.calculate_node_degree(G_deg3_4)
        nodes, edges = hp_rw.nx_to_gdf(G_deg3_4)
        nodes['x'] = nodes.geometry.centroid.x
        nodes['y'] = nodes.geometry.centroid.y

        # ===================================================================
        # ----Create city blocks (urban structure units)
        # ===================================================================
        gdf_street_raw = gdf_roads
        gdf_street_raw = gdf_street_raw.loc[~gdf_street_raw['tags.highway'].isin(['footway', 'service'])]
        city_blocks = hp_net.create_city_blocks(gdf_street_raw)

        # If cadastre, take the cadastre plots
        path_cadastre = os.path.join(path_in, "gdf_cadastre_very_large.shp")
        if type_cadastre and os.path.exists(path_cadastre):
            gdf_other_plots = gpd.read_file(path_cadastre)
            city_blocks = hp_net.clean_city_blocks(city_blocks, gdf_other_plots)
            #city_blocks.to_file(os.path.join(path_blocks, "gdf_cleaned_city_blocks.shp"))

        #city_blocks.to_file(os.path.join(path_blocks, "gdf_cleaned_city_blocks.shp"))
        city_blocks.to_postgis("gdf_cleaned_city_blocks", postgis_connection, if_exists="replace")

        # ===================================================================
        # Assumptions
        # ====================================================================
        G_crit = G.copy()
        G_deg3_4_crit = G_deg3_4.copy()
        for crit_deviation in crit_deviations:
            #path_out_characterized = os.path.join(path_temp_classified, 'characterized_edges_{}.shp'.format(crit_deviation))

            #if not os.path.exists(path_out_characterized) or write_anyway:

            assumptions = hp_sc.get_superblock_assumptions(crit_deviation) # Get assumptions

            # ====================================================================
            # Mini-S-block (3 blocks, 2x1, similar to 2x2 block)
            # ====================================================================
            id_cnt_initial = 100
            max_id, _, G_miniblockS, container_miniblocks = hp_sc.generate_miniblock_scenarios(
                G_crit,
                G_deg3_4_crit,
                tag_id_miniS,
                id_cnt_initial=id_cnt_initial,
                max_l_inner_streets=assumptions['miniblock_max_l_inner_streets'],
                max_l_outer=assumptions['miniblock_max_l_outer'],
                min_l_outer=assumptions['miniblock_min_l_outer'] ,
                min_l_inner_streets=assumptions['miniblock_min_l_inner_streets'],
                max_block_area=assumptions['miniblock_area'],
                crit_nr_nodes_deg34=assumptions['miniblock_crit_nr_nodes_deg34'],
                crit_min_pop_den=assumptions['crit_min_pop_den'],
                crit_min_GFA_den=assumptions['crit_min_GFA_den'],
                degree_nr=3,
                crit_bus_is_big_street=crit_bus_is_big_street,
                big_road_labels=big_road_labels)

            path_yaml_miniblock = os.path.join(path_temp_classified, 'characterized_edges_{}_miniblock.txt'.format(crit_deviation))
            file = open(path_yaml_miniblock, "w")
            yaml.dump(container_miniblocks, file)
            file.close()

            # Transfer attribute to street graph
            nx.set_edge_attributes(G_crit, None, label_miniS)
            nx.set_edge_attributes(G_crit, None, tag_id_miniS)
            for edge in G_miniblockS.edges:
                G_crit.edges[edge][label_miniS] = 1
                G_crit.edges[edge][tag_id_miniS] = G_miniblockS.edges[edge][tag_id_miniS]
                G_crit.edges[edge]['inter_typ'] = G_miniblockS.edges[edge]['inter_typ']

            id_cnt_initial = max_id + 100
            # ====================================================================
            # Miniblocks (2x2 block)
            # ====================================================================
            max_id, _, G_miniblock, container_miniblockS = hp_sc.generate_miniblock_scenarios(
                G_crit,
                G_deg3_4_crit,
                tag_id_minib,
                id_cnt_initial=id_cnt_initial,
                max_l_inner_streets=assumptions['miniblock_max_l_inner_streets'],
                max_l_outer=assumptions['miniblock_max_l_outer'],
                min_l_outer=assumptions['miniblock_min_l_outer'] ,
                min_l_inner_streets=assumptions['miniblock_min_l_inner_streets'],
                max_block_area=assumptions['miniblock_area'],
                crit_nr_nodes_deg34=assumptions['miniblock_crit_nr_nodes_deg34'],
                degree_nr=4,
                crit_min_pop_den=assumptions['crit_min_pop_den'],
                crit_min_GFA_den=assumptions['crit_min_GFA_den'],
                big_road_labels=big_road_labels)

            path_yaml_miniblockS = os.path.join(path_temp_classified, 'characterized_edges_{}_miniblockS.txt'.format(crit_deviation))
            file = open(path_yaml_miniblockS, "w")
            yaml.dump(container_miniblockS, file)
            file.close()

            # Transfer attribute to street graph
            nx.set_edge_attributes(G_crit, None, label_mini)
            nx.set_edge_attributes(G_crit, None, tag_id_minib)
            for edge in G_miniblock.edges:
                G_crit.edges[edge][label_mini] = 1
                G_crit.edges[edge][tag_id_minib] = G_miniblock.edges[edge][tag_id_minib]
                G_crit.edges[edge]['inter_typ'] = G_miniblock.edges[edge]['inter_typ']

            id_cnt_initial = max_id + 100
            # ====================================================================
            # Superblocks (3x3)
            # ====================================================================
            _, G_superblock, container_blocks = hp_sc.generate_superblock_scenarios(
                G_crit,
                G_deg3_4_crit,
                nodes_to_ignore=nodes_to_ignore,
                label=tag_id_superb,
                id_cnt_initial=id_cnt_initial,
                max_l_inner=assumptions['superblock_max_l_inner'],
                min_l_inner=assumptions['superblock_min_l_inner'],
                max_l_outer=assumptions['superblock_max_l_outer'],
                min_l_outer=assumptions['superblock_min_l_outer'],
                max_block_area=assumptions['superblock_area'],
                crit_nr_nodes_deg34=assumptions['superblock_crit_nr_nodes_deg34'],
                crit_min_pop_den=assumptions['crit_min_pop_den'],
                crit_min_GFA_den=assumptions['crit_min_GFA_den'],
                degree_nr=4,
                big_road_labels=big_road_labels,
                crit_bus_is_big_street=crit_bus_is_big_street)

            path_yaml_superblock = os.path.join(path_temp_classified, 'characterized_edges_{}_superblock.txt'.format(crit_deviation))
            file = open(path_yaml_superblock, "w")
            yaml.dump(container_blocks, file)
            file.close()

            # Transfer attribute to street graph
            nx.set_edge_attributes(G_crit, None, label_superb)
            nx.set_edge_attributes(G_crit, None, tag_id_superb)
            for edge in G_superblock.edges:
                G_crit.edges[edge][label_superb] = 1
                G_crit.edges[edge][tag_id_superb] = G_superblock.edges[edge][tag_id_superb]
                G_crit.edges[edge]['inter_typ'] = G_superblock.edges[edge]['inter_typ']

            logging.info("... writing out characterized")
            nodes, edges = hp_rw.nx_to_gdf(G_crit)
            #edges.to_file(path_out_characterized)
            edges.to_postgis(f"characterized_edges_{crit_deviation}", postgis_connection, if_exists="replace")

    # ---------------------------------------------------
    # And which are secondary and tertiary
    # ---------------------------------------------------
    if final_classification:
        print("... final classification")

        # Original street geometry with all original streets
        gdf_street = gpd.read_postgis(f"SELECT * FROM simplified_edges", postgis_connection, geom_col="geometry")
        G_street = hp_rw.gdf_to_nx(gdf_street, type='Graph')

        for crit_deviation in crit_deviations:

            # Load classification
            superblock_class = gpd.read_postgis(f'SELECT * FROM "characterized_edges_{crit_deviation}"', postgis_connection, geom_col="geometry")
            G_classified = hp_rw.gdf_to_nx(superblock_class)

            # Unsimplify, because for previous analysis, some edges were summarised
            G_classified = hp_net.unsimplify(G_classified, G_street)

            # ---Copy attributes from classiifcation network to full complete simplified network
            attributes_to_copy = [label_superb, label_mini, label_miniS, tag_id_superb, tag_id_minib, tag_id_miniS, 'inter_typ']
            for attribute in attributes_to_copy:
                nx.set_edge_attributes(G_street, None, attribute)

            for edge in G_classified.edges:
                for attribute in attributes_to_copy:
                    G_street.edges[edge][attribute] = G_classified.edges[edge][attribute]

            # ---Create unique classiifcation attribute. Classify roads
            G_street = hp_net.create_final_classification(
                G=G_street,
                pedestrianstreets=['pedestrian', 'living_street'],
                big_road_labels=big_road_labels,
                crit_consider_bus=crit_bus_is_big_street)

            nodes, edges = hp_rw.nx_to_gdf(G_street)
            #edges.to_file(os.path.join(path_temp_classified, 'classified_edges_{}.shp'.format(crit_deviation)))
            edges.to_postgis(f"classified_edges_{crit_deviation}", postgis_connection, if_exists="replace")

    # -------------------------------------------------------------
    # Generate individual superblocks (ranking, superblocsk first, then miniblocks)
    # -------------------------------------------------------------
    for crit_deviation in crit_deviations:
        assumptions = hp_sc.get_superblock_assumptions(crit_deviation)

        # Execute flow algorithm
        if execute_flow:
            print(".... exeucting flow")
            flow_indep.execute_flow(city, crit_deviations, path_results, path_city_raw)

        # ===================================================
        # Write intervention to folders with area from cadastre or generic
        # ===================================================
        if write_interventions and crit_deviation == write_out_scenario:
            """Write all interventions

            1. Superblocks
            2. Miniblocks
            """
            print("... write intervention")
            path_cadastre = os.path.join(path_in, 'cadastre.shp')
            if type_cadastre and os.path.exists(path_cadastre):
                gdf_cadastre_no_buildings = gpd.read_file(os.path.join(path_in, "gdf_cadastre_no_buildings.shp"))
                gdf_cadastre_no_buildings = gdf_cadastre_no_buildings.reset_index(drop=True)
                rTree_cadastre = hp_net.build_rTree(gdf_cadastre_no_buildings)
            else:
                type_cadastre = False

            # ----Load datasets
            city_blocks = gpd.read_postgis("SELECT * FROM gdf_cleaned_city_blocks", postgis_connection, geom_col="geometry")
            gdf_street = gpd.read_postgis(f'SELECT * FROM "characterized_edges_{crit_deviation}"', postgis_connection, geom_col="geometry")
            buildings = gpd.read_postgis("SELECT * FROM buildings", postgis_connection, geom_col="geometry")
            G_street = hp_rw.gdf_to_nx(gdf_street)

            path_scenario_devation = os.path.join(path_blocks, str(crit_deviation))
            hp_rw.create_folder(path_scenario_devation)

            file_container = [
                [os.path.join(path_temp, 'streets_classified', 'characterized_edges_{}_superblock.txt'.format(crit_deviation)), 'superblock', tag_id_superb],
                [os.path.join(path_temp, 'streets_classified', 'characterized_edges_{}_miniblock.txt'.format(crit_deviation)), 'miniblock', tag_id_minib],
                [os.path.join(path_temp, 'streets_classified', 'characterized_edges_{}_miniblockS.txt'.format(crit_deviation)), 'miniblockS', tag_id_miniS]]

            # Collect all implementations
            nodes_implemented = nx.Graph()
            nodes_implemented.graph['crs'] = G_street.graph['crs']

            gdf_street_areas_all = gpd.GeoDataFrame(crs=city_blocks.crs, geometry=[])
            intersect_buildings = gpd.GeoDataFrame(crs=city_blocks.crs, geometry=[])
            all_blocks = gpd.GeoDataFrame(crs=city_blocks.crs, geometry=[])
            blocks_no_street_all = gpd.GeoDataFrame(crs=city_blocks.crs, geometry=[])

            # 1. Remove cycles, find of all nodes with degree == 1 cosest insterction within buffer. add edge
            for file_path, partfilename, tag_id in file_container:
                print("Information: {} {} {}".format(file_path, partfilename, tag_id))

                # Get intervention from yaml list
                if os.path.exists(file_path):
                    with open(file_path) as file:
                        yaml_superblock = yaml.load(file, Loader=yaml.FullLoader)
                    interventions_dict = hp_net.read_ids_and_network(yaml_superblock, G_street)

                    # All intervention ids
                    inter_ids = interventions_dict.keys()

                    if mode_write_individual:
                        path_out_folder = os.path.join(path_scenario_devation, partfilename)
                        hp_rw.create_folder(path_out_folder)

                    nodes_implemented_type = nx.Graph()
                    nodes_implemented_type.graph['crs'] = G_street.graph['crs']

                    for inter_id in inter_ids:
                        print("---------Writing out: {} {}".format(tag_id, inter_id))

                        intervention_G = interventions_dict[inter_id]
                        _, intervention_gdf = hp_rw.nx_to_gdf(intervention_G)
                        assert intervention_gdf.shape[0] >= 3  # Needs minimum 3 edges

                        # Check if intersects or contained
                        type_intervention = 'full'
                        intersects_crit = False
                        contained_crit = True
                        for edge in intervention_G.edges:
                            if edge in nodes_implemented.edges:
                                intersects_crit = True
                                type_intervention = 'partial'
                            else:
                                contained_crit = False

                        # --- Graph with all implemented nodes
                        nodes_implemented = nx.compose(nodes_implemented, intervention_G)
                        nodes_implemented_type = nx.compose(nodes_implemented, intervention_G)

                        # ---Create superblock polygon based on urban structure units
                        superblock_complete = hp_net.spatial_select_fullblock(intervention_gdf, city_blocks)

                        # If contained in previuos constellation, do not write out
                        if (crit_only_full and type_intervention == 'partial') or contained_crit:  
                            #print("Intersects and not considererd")
                            pass
                        else:
                            # Check whether generated block is too big
                            block_circumference = sum(superblock_complete.geometry.length)
                            block_area = sum(superblock_complete.geometry.area)

                            if partfilename == 'superblock':
                                max_block_circumference = assumptions['superblock_max_l_outer']
                                max_block_area = assumptions['superblock_area']
                            if partfilename == 'miniblock' or partfilename == 'miniblockS':
                                max_block_circumference = assumptions['miniblock_max_l_outer']
                                max_block_area = assumptions['miniblock_area']

                            crit_fullfilled = True
                            if (block_circumference > max_block_circumference) or (block_area > max_block_area):
                                crit_fullfilled = False

                            if not crit_fullfilled or superblock_complete.shape[0] == 0 or (
                                crit_only_full and type_intervention == 'partial'):
                                    #print("Criteria not fulfilled")
                                    continue
                            else:
                                print("Write ID: {} {} {}".format(city, inter_id, partfilename))
                                if mode_write_individual:
                                    path_streets_superblock = os.path.join(path_out_folder, "{}__{}.shp".format(partfilename, inter_id))
                                    path_superblock = os.path.join(path_out_folder, "{}__{}.shp".format("block_{}".format(partfilename), inter_id))
                                    path_block_negative = os.path.join(path_out_folder, "{}__{}.shp".format("negative", inter_id))
                                    path_superblock_buildigns = os.path.join(path_out_folder, "{}__{}.shp".format("buildings", inter_id))

                                if crit_write_street_area:
                                    if type_cadastre:
                                        # ------------------------------------------------
                                        # If working with existing cadastre plots
                                        # Select cadastre based on streets of superblocks,
                                        # Split cadastre with end points of street edges
                                        # ------------------------------------------------
                                        # Buffer streets and clip cadastre files (external extrem clipping)
                                        large_buffer_dist = 30   # m
                                        gdf_cadastre_clipped = hp_osm.clip_cadastre_too_far_away(
                                            intervention_gdf, gdf_cadastre_no_buildings, large_buffer_dist, rTree_cadastre)
                                        gdf_cadastre_clipped = hp_net.gdf_multipolygon_to_polygon(gdf_cadastre_clipped)

                                        # Use clipper lines to segment cadastre
                                        offset_distance = 30            # [m] Offset distance of cutter lines (15)
                                        crit_intersection_distance = 5  # [m] Minimum intersection distance necessary to cut polygon
                                        gdf_cadastre_clipped = hp_osm.spatially_refine_cadastre_with_osm(
                                            intervention_gdf,
                                            intervention_G,
                                            gdf_cadastre_clipped,
                                            offset_distance=offset_distance,
                                            crit_intersection_distance=crit_intersection_distance)
                                        rTree_cadastre_clipped = index.Index()
                                        for edge_nr, cadastre_plot in enumerate(gdf_cadastre_clipped.geometry):
                                            rTree_cadastre_clipped.insert(edge_nr, cadastre_plot.bounds)

                                        # --Write street polygon
                                        gdf_street_areas = hp_osm.spatial_select_cadastre(
                                            intervention_gdf,
                                            gdf_cadastre_clipped,
                                            rTree_cadastre_clipped,
                                            crit_length_intersection=0.5,
                                            buffer_size=20,
                                            crit_area_intersection=0.7,
                                            min_area_of_cadastre_to_check_intersection=30)  # How much of cadastre area needs to intersect

                                        # Clip street with superblock polyon
                                        gdf_street_areas_negative = hp_net.clip_streets(superblock_complete, gdf_street_areas)
                                        gdf_street_areas = hp_net.clip_streets(superblock_complete, gdf_street_areas_negative)
                                        gdf_street_areas['geometry'] = gdf_street_areas.geometry.buffer(-0.1)
                                        gdf_street_areas['geometry'] = gdf_street_areas.geometry.buffer(0.1)
                                    else:
                                        # Get area based on street estimation
                                        print("Get area based on street estimation")
                                        gdf_street_areas = hp_net.create_artifical_street_area(intervention_gdf, buildings)
                                        gdf_street_areas['geometry'] = gdf_street_areas.buffer(0.1).buffer(-0.2).buffer(0.1)
                                        gdf_street_areas = gdf_street_areas[~gdf_street_areas.is_empty] # gdf_street_areas = gdf_street_areas.loc[gdf_street_areas.geometry > 0]

                                    # ------Write out street area
                                    if gdf_street_areas.shape[0] > 0:
                                        gdf_street_areas['b_type'] = partfilename
                                        gdf_street_areas['inter_id'] = inter_id
                                        if mode_write_individual:
                                            gdf_street_areas.to_file(path_streets_superblock)
                                        gdf_street_areas_all = gdf_street_areas_all.append(gdf_street_areas, ignore_index=True)
                                    else:
                                        raise Exception("WARNING: No superblock area_ {}".format(inter_id))

                                if crit_write_buildings:  # ---Write out buildings
                                    intersecting_buildings = hp_net.get_intersecting_buildings(buildings, superblock_complete)
                                    if len(intersecting_buildings) > 0:
                                        intersecting_buildings['b_type'] = partfilename
                                        intersecting_buildings['inter_id'] = inter_id
                                        if mode_write_individual:
                                            intersecting_buildings.to_file(path_superblock_buildigns)
                                        intersect_buildings = intersect_buildings.append(intersecting_buildings, ignore_index=True)

                                # --Write entire block
                                blocks_complete_merge = hp_net.remove_holes(superblock_complete)
                                blocks_complete_merge['b_type'] = partfilename
                                blocks_complete_merge['inter_id'] = inter_id
                                if mode_write_individual:
                                    blocks_complete_merge.to_file(path_superblock)
                                all_blocks = all_blocks.append(blocks_complete_merge, ignore_index=True)

                                if crit_write_street_negative:  # --Write negative (clip street from superblock)
                                    blocks_no_street = hp_net.clip_streets(blocks_complete_merge, gdf_street_areas)
                                    blocks_no_street['b_type'] = partfilename
                                    blocks_no_street['inter_id'] = inter_id
                                    if mode_write_individual:
                                        blocks_no_street.to_file(path_block_negative) 
                                    blocks_no_street_all = blocks_no_street_all.append(blocks_no_street, ignore_index=True)

            # Write merged shapefiles
            print("Writing out final aggregated files")
            print(gdf_street_areas_all.shape)
            print(intersect_buildings.shape)
            print(all_blocks.shape)
            print(blocks_no_street_all.shape)

            if gdf_street_areas_all.shape[0] > 0:
                gdf_street_areas_all.to_file(os.path.join(path_scenario_devation, "street_all.shp"))
                gdf_street_areas_all.to_postgis("results_street_all", postgis_connection, if_exists="replace")

            if intersect_buildings.shape[0] > 0:
                intersect_buildings.to_file(os.path.join(path_scenario_devation, "buildings_all.shp"))
                intersect_buildings.to_postgis("results_buildings_all", postgis_connection, if_exists="replace")

            if all_blocks.shape[0] > 0:
                all_blocks['area'] = all_blocks.geometry.area
                all_blocks.to_file(os.path.join(path_scenario_devation, "block_all.shp"))
                all_blocks.to_postgis("results_block_all", postgis_connection, if_exists="replace")

            if blocks_no_street_all.shape[0] > 0:
                blocks_no_street_all.to_file(os.path.join(path_scenario_devation, "negative_all.shp"))
                blocks_no_street_all.to_postgis("results_negative_all", postgis_connection, if_exists="replace")

print("_______  finished_______")
