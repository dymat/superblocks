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

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_osm as hp_osm
from superblocks.scripts.network import helper_network as hp_net


def execute_flow(city, crit_deviations, path_results, path_city_raw):
    """Edmond-Karps
    """
    calculate_base_flow = True
    transfer_attributes = True

    # Paths
    path_temp = os.path.join(path_results, str(city))
    path_intervention = os.path.join(path_temp, 'cleaned_edges.shp')
    path_bb = os.path.join(path_city_raw, str(city), "extent.shp")
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
        bb_shp = gpd.read_file(path_bb)
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
        gdf_street = gpd.read_file(path_intervention)

        # Remove some street types
        other_streets = ['pedestrian', 'living_street']
        gdf_street = gdf_street.loc[~gdf_street['tags.highw'].isin(other_streets)]

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
        edges.to_file(os.path.join(path_out, "G_base_flow.shp"))

        # Weigh local and global flow
        G_global = hp_net.flow_reg_glob_calc(
            G_base_flow, f_local=f_local, f_gobal=f_gobal, label_one='local_p', label_two='glob_p', label='flow_ov')

        _, gdf_global = hp_rw.nx_to_gdf(G_global)
        gdf_global.to_file(os.path.join(path_out, "base_flow_edmunds_recalculated.shp"))


    # ------------------------------------------------------------------------
    # Transfer attributes across diferent deviation scenarios
    # ------------------------------------------------------------------------$
    if transfer_attributes:
        gdf_global = gpd.read_file(os.path.join(path_out, "base_flow_edmunds_recalculated.shp"))
        G_global = hp_rw.gdf_to_nx(gdf_global)

        for crit_deviation in crit_deviations:
            #print("... copying attributes to file {}".format(crit_deviation))
            # Load file with crit dev results

            gdf_destination = gpd.read_file(os.path.join(
                path_temp, 'streets_classified', "classified_edges_{}.shp".format(crit_deviation)))
            path_classified = os.path.join(path_out, "classified_edges_withflow_{}.shp".format(crit_deviation))

            G_destination = hp_rw.gdf_to_nx(gdf_destination)
            for edge in G_global.edges:
                if edge in G_destination.edges:
                    G_destination.edges[edge]['flow_ov'] = G_global.edges[edge]['flow_ov']

            _, edges = hp_rw.nx_to_gdf(G_destination)
            edges.to_file(path_classified)


# ------------------------------------------------------------
# Sole mode (___main____)
# ------------------------------------------------------------
sole_mode = False
if sole_mode:
    crit_deviations = [2]
    execute_flow(city='Rome', crit_deviations=crit_deviations)

# ------------------------------------------------------------
# Intervention flow
# ------------------------------------------------------------
plot_intervention_average_flow = False
if plot_intervention_average_flow:

    # ==============================================================
    # Simple plot where the sum of the normalized flow of each
    # edge of an intervention is plotted
    # ==============================================================
    result_conatiner = []
    label = 'flow_ov'
    length_factor = 1000  # [m] to km
    for inter_id in inter_ids:
        intervention_gdf = gdf_global.loc[gdf_global[tag_id] == inter_id]
        tot_length = sum(intervention_gdf.geometry.length)

        # Normalized average value
        summed_value = 0
        for loc in intervention_gdf.index:
            summed_value += intervention_gdf.loc[loc][label] * intervention_gdf.loc[loc]['geometry'].length
        averaged_value = round(summed_value / tot_length, 5)
        result_conatiner.append((inter_id, tot_length / length_factor, averaged_value))

    pd_results = pd.DataFrame(result_conatiner, columns=['id', 'length', 'av_flow_disruption'])

    # Plot scatter plot with intervention length versus average disruption
    pd_results1 = pd_results.set_index('id', drop=True)
    pd_results1 = pd_results1.sort_values(by=['length'])
    pd_results1.plot(x='length', y='av_flow_disruption', kind='scatter')
    plt.xlabel("length intervention [km]")
    plt.ylabel("av_flow_disruption [-]")
    plt.show()

    # Plot line
    pd_results2 = pd_results.sort_values(by=['av_flow_disruption'])
    pd_results2.reset_index(drop=True).plot(y='av_flow_disruption', kind='line')
    plt.xlabel('index column, (not id)')
    plt.ylabel('av_flow_disruption')
    plt.show()
    pd_results3 = pd_results.sort_values(by=['length'])
    pd_results3.reset_index(drop=True).plot(y='length', kind='scatter')
    plt.xlabel('index column, (not id)')
    plt.ylabel('length')
    plt.show()

# ==============================================================
# 
# Calculate flows per iteration
# 
# ==============================================================
def calculate_flow_iterateions():
    result_conatiner = []
    for inter_id in inter_ids:

        G_copy = G.copy()

        # ---Get single intervention
        intervention_gdf = gdf_global.loc[gdf_global[tag_id] == inter_id]
        G_intervention = hp_rw.gdf_to_nx(intervention_gdf)
        intervention_gdf.to_file(os.path.join(path_out, "intervention_{}.shp".format(inter_id)))

        # ----Remove intervention from network
        G_copy.remove_edges_from(G_intervention.edges)

        # -----------------------
        # ----Run local algorithm
        # -----------------------
        nx.set_edge_attributes(G_copy, 0, 'local_av')
        nx.set_edge_attributes(G_copy, 0, 'local_flow')
        nx.set_edge_attributes(G_copy, 0, 'local_p')
        for local_bb in local_bbs:
            local_bb_obj = hp_osm.BB(ymax=max(local_bb.bounds.maxy), ymin=min(local_bb.bounds.miny), xmax=max(local_bb.bounds.maxx), xmin=min(local_bb.bounds.minx))

            # Get local streets
            G_local = hp_net.get_intersecting_edges(G_copy, bb=local_bb)

            # Create helper netowrk
            G_local = flow_algorithm_functions.clean_network_and_assign_capacity(G_local, flow_factor=flow_factor)

            # Calcualte flow
            G_local, dict_G_super_local = flow_algorithm_functions.create_super_sinks_sources(
                G_local, local_bb_obj, nr_help_pnts=nr_help_pnts)

            G_local = flow_algorithm_functions.flow_emund(
                G_local, dict_G_super_local, max_road_cap=max_road_cap)

            # Pass local flow to full network
            for edge_local in G_local.edges:
                G_copy.edges[edge_local]['local_av'] = G_base_flow.edges[edge_local]['av_flow']
                G_copy.edges[edge_local]['local_flow'] = G_local.edges[edge_local]['sum_flow']
                G_copy.edges[edge_local]['local_p'] = G_local.edges[edge_local]['rel_flow']

        # ------------------------------
        # ----Run global flow algorithm
        # ------------------------------
        G_super, dict_G_super = flow_algorithm_functions.create_super_sinks_sources(
            G_copy, bb_global, nr_help_pnts=nr_help_pnts)

        G_flow = flow_algorithm_functions.flow_emund(
            G_super, dict_G_super, max_road_cap=max_road_cap)

        # Write global flow
        nx.set_edge_attributes(G_flow, 0, 'glob_flow')
        nx.set_edge_attributes(G_flow, 0, 'glob_p')
        for edge_local in G_flow.edges:
            G_flow.edges[edge_local]['glob_flow'] = G_flow.edges[edge_local]['sum_flow']
            G_flow.edges[edge_local]['glob_p'] = G_flow.edges[edge_local]['rel_flow']

        # Calculate global_local relationship of flow
        G_flow = hp_net.flow_reg_glob_calc(G_flow, f_local=f_gobal, f_gobal=f_gobal, label='flow_ov')
        #nodes, edges = hp_rw.nx_to_gdf(G_flow)
        #edges.to_file("C:/_scrap/G_flow.shp")
        #nodes, edges = hp_rw.nx_to_gdf(G_global)
        #edges.to_file("C:/_scrap/G.shp")

        # ---- Calculate flow difference (absoltue and in percent)
        G_flow, norm_diff, _ = flow_algorithm_functions.calc_flow_difference(
            G_global, G_flow, attribute='local_p', label='d_localp')

        G_flow, norm_diff, _ = flow_algorithm_functions.calc_flow_difference(
            G_global, G_flow, attribute='flow_ov', label='d_flowov')

        G_flow, norm_diff, _ = flow_algorithm_functions.calc_flow_difference(
            G_global, G_flow, attribute='centrality', label='d_central')


        nodes, edges = hp_rw.nx_to_gdf(G_flow)
        edges.to_file(os.path.join(path_out, "edges_G_diff.shp"))
        prnt(":")
        result_conatiner.append(["id_{}".format(inter_id), norm_diff])

    fig, ax = plt.subplots(figsize=hp_rw.cm2inch(9, 9))
    df_raw = pd.DataFrame(result_conatiner, columns=['id_superblock', 'norm_diff'])
    df_raw = df_raw.set_index('id_superblock')
    df_raw = df_raw.sort_values(by=['norm_diff'], ascending=False)
    df_raw.to_csv("C:/_scrap/results.csv")
    df_raw.plot(y='area', kind='bar', color='grey', ax=ax)
    plt.show()



'''
# Run flow algorithm
G = flow_algorithm_functions.flow_emund(G, dict_G_super, max_road_cap=10)

nodes, edges = hp_rw.nx_to_gdf(G)
nodes.to_file(os.path.join(path_out, "nodes_flow.shp"))
edges.to_file(os.path.join(path_out, "edges_flow.shp"))
'''