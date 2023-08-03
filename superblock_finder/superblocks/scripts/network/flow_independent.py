"""
[  ] 

"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd

from sqlalchemy import create_engine

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_osm as hp_osm
from superblocks.scripts.network import helper_network as hp_net

postgis_connection = create_engine(f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
                                   f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}" \
                                   f"@{os.getenv('POSTGRES_HOST', 'localhost')}:5432/{os.getenv('POSTGRES_DATABASE', 'postgres')}")



def execute_flow(city, crit_deviation, path_results, path_city_raw):
    """Edmond-Karps
    """
    calculate_base_flow = True
    transfer_attributes = True

    # Paths
    path_temp = os.path.join(path_results, str(city))
    flow_folder = '_flows'
    path_out = os.path.join(path_temp, flow_folder)
    hp_rw.create_folder(path_out)

    tag_id = 'id_superb'
    nr_help_pnts = 100
    flow_factor = 1

    # Weighting factors or flow_ov
    f_local = 1
    f_gobal = 1

    if calculate_base_flow:
        # =============================================================================================
        # Load bounding box
        # =============================================================================================
        #bb_shp = gpd.read_file(path_bb)
        bb_shp = gpd.read_postgis("SELECT * FROM bbox", postgis_connection, geom_col="geometry")
        bb_global = hp_osm.BB(
            ymax=max(bb_shp.bounds.maxy), ymin=min(bb_shp.bounds.miny),
            xmax=max(bb_shp.bounds.maxx), xmin=min(bb_shp.bounds.minx))
        bb_shp = bb_global.as_gdf(crs_orig=int(bb_shp.crs.srs.split(":")[1]))

        # =============================================================================================
        # Create local flow calculation bboxes
        # =============================================================================================
        fishnet_size = 2000  # [m] Size of fishgrid
        regional_bb = bb_shp
        local_bbs = hp_net.generate_fishnet(regional_bb, crit_bb_width=fishnet_size)

        # Load data
        #gdf_street = gpd.read_file(path_intervention)
        gdf_street = gpd.read_postgis("SELECT * FROM cleaned_edges", postgis_connection, geom_col="geometry")

        # Remove some street types
        other_streets = ['pedestrian', 'living_street']
        gdf_street = gdf_street.loc[~gdf_street['tags.highway'].isin(other_streets)]

        # Full network
        G = hp_rw.gdf_to_nx(gdf_street)
        #TODO: aug2021 removed gdf_street = gdf_street.loc[~gdf_street[tag_id].isna()] # Only interventions

        # =============================================================================================
        # Calculate max flow capacity
        # =============================================================================================
        G = flow_algorithm_functions.clean_network_and_assign_capacity(G, flow_factor=flow_factor)
        max_flow = 0
        for edge in G.edges:
            capacity = G.edges[edge]['capacity']
            if capacity > max_flow:
                max_flow = capacity
        max_road_cap = max_flow

        # =============================================================================================
        # Calculate local flow by interating over local windows
        # =============================================================================================
        nx.set_edge_attributes(G, 0, 'local_flow')
        nx.set_edge_attributes(G, 0, 'local_av')
        nx.set_edge_attributes(G, 0, 'local_p')

        for local_bb in local_bbs:
            local_bb_obj = hp_osm.BB(ymax=max(local_bb.bounds.maxy), ymin=min(local_bb.bounds.miny), xmax=max(local_bb.bounds.maxx), xmin=min(local_bb.bounds.minx))

            # Get local streets
            G_local = hp_net.get_intersecting_edges(G, bb=local_bb)
            number_of_edges = G_local.number_of_edges()
            print("Number of edges: {}".format(G_local.number_of_edges()))
            if number_of_edges == 0:
                pass
            else:
                _, edges = hp_rw.nx_to_gdf(G_local)

                # Create helper netowrk
                G_local = flow_algorithm_functions.clean_network_and_assign_capacity(G_local, flow_factor=flow_factor)

                # Calcualte flow
                G_local, dict_G_super_local = flow_algorithm_functions.create_super_sinks_sources(
                    G_local, local_bb_obj, nr_help_pnts=nr_help_pnts)

                G_base_flow = flow_algorithm_functions.flow_emund(
                    G_local, dict_G_super_local, max_road_cap=max_road_cap)

                # Pass local flow to full network
                for edge_local in G_base_flow.edges:
                    G.edges[edge_local]['local_flow'] = G_base_flow.edges[edge_local]['sum_flow']
                    G.edges[edge_local]['local_av'] = G_base_flow.edges[edge_local]['av_flow']
                    G.edges[edge_local]['local_p'] = G_base_flow.edges[edge_local]['rel_flow']

        # =============================================================================================
        # Measure calculations
        # m_flow: Original flow
        # d_flow: Change of overall network flow due to intervention (change in percentage of each edge)
        # coverg: Versiegelungsgrad innheralb eines bestimmten Radius
        # =============================================================================================
        inter_ids = []
        for i in list(set(gdf_street[tag_id])):
            if i != None:
                inter_ids.append(int(i))

        # =============================================================================================
        # Global flow algorithm
        # =============================================================================================
        G = flow_algorithm_functions.clean_network_and_assign_capacity(G, flow_factor=flow_factor)
        #nodes, edges = hp_rw.nx_to_gdf(G)
        #edges.to_file(os.path.join(path_out, "capacity_base.shp"))

        # Create supergraph for flow algorithm
        G_super, dict_G_super = flow_algorithm_functions.create_super_sinks_sources(
            G, bb_global, nr_help_pnts=nr_help_pnts)

        # Calculate base flow
        G_base_flow = flow_algorithm_functions.flow_emund(
            G_super, dict_G_super, max_road_cap=max_road_cap)

        # Write global flow
        nx.set_edge_attributes(G, 0, 'glob_flow')
        nx.set_edge_attributes(G, 0, 'glob_av')
        nx.set_edge_attributes(G, 0, 'glob_p')

        for edge_local in G_base_flow.edges:
            G_base_flow.edges[edge_local]['glob_flow'] = G_base_flow.edges[edge_local]['sum_flow']
            G_base_flow.edges[edge_local]['glob_av'] = G_base_flow.edges[edge_local]['av_flow']
            G_base_flow.edges[edge_local]['glob_p'] = G_base_flow.edges[edge_local]['rel_flow']

        _, edges = hp_rw.nx_to_gdf(G_base_flow)
        #edges.to_file(os.path.join(path_out, "G_base_flow.shp"))
        edges.to_postgis("G_base_flow", postgis_connection, if_exists="replace")

        # Weigh local and global flow
        G_global = hp_net.flow_reg_glob_calc(
            G_base_flow, f_local=f_local, f_gobal=f_gobal, label_one='local_p', label_two='glob_p', label='flow_ov')

        _, gdf_global = hp_rw.nx_to_gdf(G_global)
        #gdf_global.to_file(os.path.join(path_out, "base_flow_edmunds_recalculated.shp"))
        gdf_global.to_postgis("base_flow_edmunds_recalculated", postgis_connection, if_exists="replace")


    # ------------------------------------------------------------------------
    # Transfer attributes across different deviation scenarios
    # ------------------------------------------------------------------------$
    if transfer_attributes:
        try:
            gdf_global
        except:
            gdf_global = gpd.read_postgis("SELECT * FROM base_flow_edmunds_recalculated", postgis_connection, geom_col="geometry")

        G_global = hp_rw.gdf_to_nx(gdf_global)

        #print("... copying attributes to file {}".format(crit_deviation))
        # Load file with crit dev results

        gdf_destination = gpd.read_postgis(f'SELECT * FROM "classified_edges_{crit_deviation}"', postgis_connection, geom_col="geometry")

        G_destination = hp_rw.gdf_to_nx(gdf_destination)
        for edge in G_global.edges:
            if edge in G_destination.edges:
                G_destination.edges[edge]['flow_ov'] = G_global.edges[edge]['flow_ov']

        _, edges = hp_rw.nx_to_gdf(G_destination)

        edges.to_postgis(f'classified_edges_withflow_{crit_deviation}', postgis_connection, if_exists="replace")


