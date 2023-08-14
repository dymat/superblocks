"""
Helper function to create scenarios
"""
import os
import sys
import logging
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
from progress.bar import Bar

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import helper_network as hp_net

def get_superblock_assumptions(p_d):
    """
    Calculate minimum and maximum length based on
    deviation of iriginal barcelonian suberlbokc
    """
    #ha_factor = 10000
    uncertainty = 0.2  # [m] 20%
    max_uncertainty = 1 + uncertainty
    min_uncertainty = 1 - uncertainty
    barcelona_width = 400  # [m] Width of oirignal superblock
    barcelona_nr_blocks = 3
    b_width = (barcelona_width / barcelona_nr_blocks)

    factor = 1 + p_d

    assumptions = {
        'crit_min_pop_den': 100,    # [pop / ha] Minimum population density
        'crit_min_GFA_den': 0.3,    # [decimal] Fraction of GVA covered

        # Area in ha Note: NEW: Overall size not larger
        #'miniblock_area': (b_width * 2 * factor) * (b_width * 2 * factor), # / ha_factor,
        #'superblock_area': (b_width * 3 * factor) * (b_width * 3 * factor), # / ha_factor,
        'miniblock_area': round((b_width * factor * 2) * (b_width * factor * 2) * max_uncertainty, 1),  # / ha_factor,
        'superblock_area': round((b_width * factor * 3) * (b_width * factor * 3) * max_uncertainty, 1),  # / ha_factor,

        # Minibloks
        'miniblock_max_l_outer': round(b_width * 8 * max_uncertainty * factor, 3),           # [m] Maximum length of outer surrounding circle
        'miniblock_min_l_outer': round(b_width * 8 * min_uncertainty / factor, 3),          # [m] Minimum length of outer surrounding circle

        # for 2x1 miniblock
        'miniblock_min_l_inner_streets': round(b_width * min_uncertainty * 3 / factor, 3),   # [m] Minimum length of all inner roads
        'miniblock_max_l_inner_streets': round(b_width * max_uncertainty * 4 * factor, 3),   # [m] Minimum length of all inner roads
        'miniblock_crit_nr_nodes_deg34': 1,                                                 # [-] Number of nodes with degree 3 or 4

        # Superblock
        'superblock_max_l_inner': round(b_width * 4 * max_uncertainty * factor, 3),   # [m] Maximum length of inner circle
        'superblock_min_l_inner': round(b_width * 4 * min_uncertainty / factor, 3),   # [m] Minimum length of inner circle
        'superblock_max_l_outer': round(b_width * 12 * max_uncertainty * factor, 3),  # [m] Maximum length of outer circle
        'superblock_min_l_outer': round(b_width * 12 * min_uncertainty / factor, 3),  # [m] Minimum length of outer circle
        'superblock_crit_nr_nodes_deg34': 3     # [-] How many nodes with neighbour creed intervention needs to have
    }

    return assumptions


def generate_miniblock_scenarios(
        G,
        G_deg3_4,
        tag_id,
        id_cnt_initial,
        max_l_inner_streets=False,
        min_l_inner_streets=False,
        max_l_outer=False,
        min_l_outer=False,
        max_block_area=False,
        crit_nr_nodes_deg34=1,
        degree_nr=4,
        crit_min_pop_den=80,
        crit_min_GFA_den=0.3,
        crit_bus_is_big_street=False,
        big_road_labels=[]
        ):
    """Select miniblocks
    """
    G_local = G.copy()
    G_miniblock = G_deg3_4.copy()
    #a, b = hp_rw.nx_to_gdf(G_miniblock)
    #b.to_file("J:/Sven/_scrap/miniblock.shp")
    # Do not consider service roads
    G_local = hp_net.remove_edge_by_attribute(G_local, attribute='tags.highway', value="service")
    G_miniblock = hp_net.remove_edge_by_attribute(G_miniblock, attribute='tags.highway', value="service")

    G_strategy = nx.Graph()
    G_strategy.graph = G_miniblock.graph

    _, lodes_larger_deg = hp_net.get_subgraph_degree(G_miniblock, degree_nr=degree_nr, method='node_removal')
    nx.set_edge_attributes(G_strategy, 0, tag_id)
    nx.set_edge_attributes(G_strategy, 0, 'inter_typ')

    bar = Bar("Miniblock selection", max=len(lodes_larger_deg))
    id_cnt = id_cnt_initial
    container_blocks = {}
    for node_4 in lodes_larger_deg:

        nodes_neighbouring = list(G_local.neighbors(node_4))
        assert len(nodes_neighbouring) >= 3, "Error: {} {} {} ".format(len(nodes_neighbouring), 3, node_4)

        # Note: Maybe sort according to distacne rather?
        nodes_geographically_sorted = hp_net.sort_geographically_clockwise(nodes_neighbouring)
        nodes_geographically_sorted.append(nodes_geographically_sorted[0])  # close loop
        assert nodes_geographically_sorted[0] == nodes_geographically_sorted[-1]

        # Create subgraph and delete subnetwork from G
        G_inner = hp_net.create_subgraph_interior_roads_miniblock(node_4, nodes_neighbouring, G_local)
        #gpd.GeoDataFrame([Point(i) for i in list(G_inner.nodes)], columns=['geometry']).to_file("C:/_scrap/inner_nodes.shp")
        #_____ = hp_rw.nx_to_gdf(G_inner)
        #for edge in G_inner.edges:
        #    if G_local.edges[edge]['tags.name'] == 'West Peachtree Street Northwest':
        #        print("--")
        G_local.remove_edges_from(G_inner.edges)

        # Calculate path grand cycle
        path_crit, outer_cycle_length, block_area_cycling_path, block = hp_net.find_grand_cycle(
            G_local, nodes_geographically_sorted, list(G_inner.nodes), relaxation_crit=True)
        #block_cycling = gpd.GeoDataFrame([block_cycling], columns=['geometry'])
        #block_cycling.to_file("C:/_scrap/A.shp")
        #gpd.GeoDataFrame([LineString(grand_cyle_path)], columns=['geometry']).to_file("C:/_scrap/grand_cyle_path.shp")
        #gpd.GeoDataFrame([Point(i) for i in list(G_inner.nodes)], columns=['geometry']).to_file("C:/_scrap/nodes.shp")

        # ----------------------------------------
        # Check if a big road or tram/bus is in block as well on inner intersection (5.10.2021)
        # ---------------------------------------------
        if path_crit: 
            G_inner_block = hp_net.get_all_edges_in_block(G_local, block)
            G_inner_block = nx.compose(G_inner_block, G_inner) #New: ADD INNTER NODE PLUS CHECK ON FULL
            for edge in G_inner_block.edges:
                if block.contains(LineString(edge)):
                    tag_highway = G.edges[edge]['tags.highway']
                    tag_tram = G.edges[edge]['tram']
                    tag_trolley_bus = G.edges[edge]['trolleybus']
                    if crit_bus_is_big_street:
                        tag_bus = G.edges[edge]['bus']
                    else:
                        tag_bus = False

                    if tag_highway in big_road_labels or tag_bus or tag_tram or tag_trolley_bus:
                        path_crit = False

        # Add removed subgraph again
        G_local = nx.compose(G_local, G_inner)

        inner_street_l = hp_net.get_length_network(G_inner)

        # Average population density of Miniblock streets
        average_pop_den = G_local.nodes[node_4]['pop_den']
        average_GFA_den = G_local.nodes[node_4]['GFA_den']

        node_degree_crit = hp_net.nr_edges_34_subgraph(G_inner, degree=degree_nr)

        min_nodes_34 = 1   # [-] Minimum number of nodes
        nr_nodes_34 = hp_net.nr_edges_34_subgraph(G_inner)

        if not path_crit or (block_area_cycling_path > max_block_area) or (inner_street_l > max_l_inner_streets) or (
            outer_cycle_length > max_l_outer) or (
                outer_cycle_length < min_l_outer) or (
                    inner_street_l < min_l_inner_streets) or (
                        node_degree_crit < crit_nr_nodes_deg34) or (
                            average_GFA_den < crit_min_GFA_den and not average_pop_den > crit_min_pop_den) or (
                                nr_nodes_34 < min_nodes_34  # NEW 3.10.2021
                                ):
            logging.info(" ")
            logging.info("--- Minibock is ignored")
            logging.info("inner_street_l:          {}".format(inner_street_l))
            logging.info("max_l_inner_streets:     {}".format(max_l_inner_streets))
            logging.info("min_l_inner_streets:     {}".format(min_l_inner_streets))
            logging.info("outer_cycle_length:      {}".format(outer_cycle_length))
            logging.info("max_l_outer:             {}".format(max_l_outer))
            logging.info("min_l_outer:             {}".format(min_l_outer))
            logging.info("path_crit:               {}".format(path_crit))
            logging.info("average_GFA_den:         {}".format(average_GFA_den))
            logging.info("average_pop_den:         {}".format(average_pop_den))
        else:
            # Get extension type #TODO: Export as function
            remove_overlappin_elements = False  # True: Deletes partial elements

            extention_type = 'full'
            cycle_nodes_to_remove = []
            for cycle_node in nodes_geographically_sorted :
                if cycle_node in G_strategy.nodes and remove_overlappin_elements:
                    cycle_nodes_to_remove.append(cycle_node)
                    extention_type = 'partial'
            edges = hp_net.to_tuple_list(nodes_geographically_sorted)

            # Add miniblock
            for edge in G_inner.edges:
                if edge[0] in cycle_nodes_to_remove and edge[1] in cycle_nodes_to_remove:  # edge was already considered
                    pass
                else:
                    #if G_strategy.has_edge(*edge): # NOTE: REMOVED AGAIN
                    #    pass #Note: added 16.08.2021. Only replace if not in full
                    #else:
                    #if True:
                    G_strategy.add_edge(edge[0], edge[1])
                    for attribute, value in G_local.edges[edge].items():
                        G_strategy.edges[edge][attribute] = value
                    G_strategy.edges[edge][tag_id] = int(id_cnt)
                    G_strategy.edges[edge]['inter_typ'] = extention_type

            # Store all edges in container
            for edge in G_inner.edges:
                if edge in container_blocks.keys():
                    container_blocks[edge].append(id_cnt)
                else:
                    container_blocks[edge] = [id_cnt]

        id_cnt += 1
        bar.next()
    bar.finish()

    if G_strategy.number_of_edges() > 0:
        nodes, edges = hp_rw.nx_to_gdf(G_strategy)
    else:
        edges = gpd.GeoDataFrame()
        logging.info("No intervention found for miniblock")

    return id_cnt, edges, G_strategy, container_blocks


def generate_superblock_scenarios(
        G,
        G_deg3_4,
        nodes_to_ignore,
        label,
        id_cnt_initial,
        max_l_inner,
        min_l_inner,
        max_l_outer,
        min_l_outer,
        max_block_area,
        crit_nr_nodes_deg34,
        crit_min_pop_den,
        crit_min_GFA_den,
        degree_nr,
        big_road_labels,
        crit_bus_is_big_street=False
    ):
    """
    Create all interventions and return as list with gdf

    Note: Depending on the order of how the 4x4 nodes are iterated, the interventions change
    """
    print("...start generate_superblock_scenarios")

    # --- Get only superblock edges
    min_deg_neig = 3
    G_local = G.copy()
    G_superblock = G_deg3_4.copy()
    G_strategy = nx.Graph()
    G_strategy.graph = G_deg3_4.graph

    nx.set_edge_attributes(G_strategy, 0, label)
    nx.set_edge_attributes(G_strategy, 0, 'inter_typ')

    # Do not consider service roads for cycles
    G_superblock = hp_net.remove_edge_by_attribute(G_superblock, attribute='tags.highway', value="service")

    # Fast heuristic to find cycles
    _, lodes_larger_deg = hp_net.get_subgraph_degree(G_superblock, degree_nr=4, method='node_removal')
    all_cycles = hp_net.get_superblock_cycles(G_superblock, lodes_larger_deg, max_l_outer)

    print("Number of cycles: {} ".format(len(all_cycles)))
    #gpd.GeoDataFrame([LineString(cycle) for cycle in all_cycles], columns=['geometry'], crs=G_deg3_4.graph['crs']).to_file(
    #    os.path.join(out_path, 'all_cycles.shp'))

    # --- Remove cycles which cross big streets
    not_good_cycles = []
    for cycle in all_cycles:
        for cycle_node in cycle:
            if cycle_node in nodes_to_ignore:
                if cycle not in not_good_cycles:
                    not_good_cycles.append(cycle)

    # --- Remove cycles with too long inner circles
    for cycle in all_cycles:
        circle_distance = hp_net.get_distance_along_path(G_local, cycle)  # length of inner circle along full network

        if (circle_distance > max_l_inner) or circle_distance < min_l_inner:
            if cycle not in not_good_cycles:
                not_good_cycles.append(cycle)

    # --- Remove cycles which cross bridg or tunnel (new: R3)
    for cycle in all_cycles:
        edges = hp_net.to_tuple_list(cycle)
        for edge in edges:
            tunnel_crit = G_local.edges[edge]['tags.tunnel']
            bridge_crit = G_local.edges[edge]['tags.bridge']
            if tunnel_crit == 1 or bridge_crit == 1:
                if cycle not in not_good_cycles:
                    not_good_cycles.append(cycle)

    # Remove cycles
    for too_long_cycle in not_good_cycles:
        all_cycles.remove(too_long_cycle)

    #gpd.GeoDataFrame([LineString(cycle) for cycle in all_cycles], columns=['geometry'], crs=G_deg3_4.graph['crs']).to_file(
    #    os.path.join(out_path, "sorted_cycles_removed_too_long_inner.shp"))

    # --- Remove cycles with no surrounding circle or with too long surrounding circle
    bar = Bar("Superblock cycle testing", max=len(all_cycles))
    not_good_cycles = []
    for cycle in all_cycles:

        # Outer cycle length, get all neighbors of cycle which are for sure connected to the wider network (not degree = 1)
        all_neighbours_of_cycle = []
        all_influcing_edges = []

        for node in cycle:
            neigh_nodes_4deg = list(G_local.neighbors(node))
            for neighbor in neigh_nodes_4deg:
                if neighbor not in cycle:
                    all_neighbours_of_cycle.append(neighbor)
            # Note all neighbours which are for sure not connected to the ntwork and thus no surrounding path
            for neighbor in neigh_nodes_4deg:
                if G_local.degree[neighbor] >= min_deg_neig and neighbor not in cycle:  # Needs to be a crossing
                    all_influcing_edges.append((node, neighbor))
        all_neighbours_of_cycle = list(set(all_neighbours_of_cycle))

        if len(all_neighbours_of_cycle) == 0:  # Strange shape
            not_good_cycles.append(cycle)
        else:
            all_neighbours_of_cycle_sorted = hp_net.sort_geographically_clockwise(all_neighbours_of_cycle)

            # Create subgraph and temporaly remove subgraph from G to calculate path
            all_nodes_to_remove = cycle + all_neighbours_of_cycle

            G_inner = hp_net.create_subgraph_interior_roads_superblock(
                cycle, all_neighbours_of_cycle, G_local)

            #if cycle == [(741565.6062684513, 3738327.5332248486), (741617.5969557862, 3738271.090665834), (741674.4847295785, 3738318.924498512), (741621.6393054437, 3738377.6314931223), (741596.2589147394, 3738354.835529504), (741565.6062684513, 3738327.5332248486)]:
            #    print("--")
            #for edge in G_inner.edges:
            #    if G_local.edges[edge]['tags.name'] == "West Peachtree Street Northwest":
            #        print("--")

            G_local.remove_edges_from(G_inner.edges)

            # Calculate block of a list of nodes
            path_crit, outer_cycle_length, block_cycling_area, block_cycling = hp_net.find_grand_cycle(
                G=G_local, all_neighbours_of_cycle=all_neighbours_of_cycle_sorted,
                inner_street=all_nodes_to_remove,
                relaxation_crit=True)

            # Get all edges within block and check whether not good street contained
            inner_road_is_too_big = False
            if path_crit:
                G_in_block = hp_net.get_all_edges_in_block(G_local, block_cycling)
                for inner_edge in G_in_block.edges:
                    tag_highway = G_in_block.edges[inner_edge]['tags.highway']
                    crit_tram = G_in_block.edges[inner_edge]['tram']
                    crit_trolleybus = G_in_block.edges[inner_edge]['trolleybus']
                    crit_bus = G_in_block.edges[inner_edge]['bus']
                    if not crit_bus_is_big_street:
                        crit_bus = False
                    if tag_highway in big_road_labels or crit_bus or crit_tram or crit_trolleybus:
                        inner_road_is_too_big = True

            if inner_road_is_too_big or not path_crit:
                not_good_cycles.append(cycle)
                G_local = nx.compose(G_local, G_inner)  # Add removed subgraph again
            else:
                G_local = nx.compose(G_local, G_inner)  # Add removed subgraph again

                # See which could potentially be a superblock based on area and circumference
                #if block_cycling_area > max_block_area and (outer_cycle_length < max_l_outer):
                #    print("WARNING: superblock: {}".format((100 / block_cycling_area) * max_block_area))

                if (block_cycling_area > max_block_area) or (
                        outer_cycle_length > max_l_outer) or (
                            outer_cycle_length < min_l_outer):
                    #print("Criteria not fulfilled: {} {} {} {}".format(path_crit, outer_cycle_length, max_l_outer, min_l_outer))
                    not_good_cycles.append(cycle)
                else:
                    #print("found a valid circle")
                    pass # good cycle

        bar.next()
    bar.finish()

    for too_long_cycle in not_good_cycles:
        all_cycles.remove(too_long_cycle)

    if len(all_cycles) > 0:
        #gpd.GeoDataFrame([LineString(cycle) for cycle in all_cycles], columns=['geometry']).to_file(os.path.join(out_path, "sorted_cycles_removed_too_long_outer_and_inner.shp"))
        pass
    else:
        print("NO MINIBLOCKS VALID ANYMORE")

    # ===============================
    # Iterate all potential cycles and select randomly node to start test
    # A distinction between full and partial implementation is possible
    # ===============================
    container_blocks = {}
    bar = Bar("Superblock cycle testing", max=len(all_cycles))

    id_cnt = id_cnt_initial
    for node_4_4 in G_superblock.nodes:
        if len(all_cycles) == 0:  # All cycles are tested
            continue
        if "degree" in G_superblock.nodes[node_4_4] and G_superblock.nodes[node_4_4]['degree'] >= 4:  # If 4x4 node, perform analysis

            # ---Find smallest cycle which contains cycle edge
            cylce_cnt = 0
            found_cycle = False
            iter_min_cycle_l = 9999999999
            for cycle in all_cycles:
                if node_4_4 in cycle: 
                    inner_cycle_length = hp_net.get_distance_along_path(G_local, cycle)
                    if inner_cycle_length < iter_min_cycle_l:
                        iter_min_cycle_l = inner_cycle_length
                        found_cycle = True
                        cycle_nodes = cycle
                        cylce_cnt += 1

            #print("Number of cycles detected: {}".format(cylce_cnt))
            if found_cycle:
                assert cycle_nodes[0] == cycle_nodes[-1]  # Check that closed

                # Remove cycles which is tested
                all_cycles.remove(cycle_nodes)
                cycle_edges = hp_net.to_tuple_list(cycle_nodes)

                # Average population density of superblock streets, but only of inner cycle
                average_pop_den = hp_net.collect_average_edge_attribute(G, edges_path=cycle_edges, attribute='pop_den', norm=True)
                average_GFA_den = hp_net.collect_average_edge_attribute(G, edges_path=cycle_edges, attribute='GFA_den', norm=True)
                node_degree_crit = hp_net.nr_edges_34_cycle(G_superblock, cycle_nodes, degree=degree_nr)

                #logging.info("------ Superblocks ------")
                #logging.info("average_pop_den:     {}".format(average_pop_den))
                #logging.info("average_GFA_den:     {}".format(average_GFA_den))
                #logging.info("node_degree_crit:    {}".format(node_degree_crit))
                # Test urban density criteria
                if (average_GFA_den > crit_min_GFA_den or average_pop_den > crit_min_pop_den) and (
                        node_degree_crit >= crit_nr_nodes_deg34):

                    # Check if only part of a superblock, e.g. attached to it as already another superblock has been implemented
                    remove_overlappin_elements = False  # True: Deletes partial elements

                    extention_type = 'full'
                    cycle_edges_to_remove = []

                    if remove_overlappin_elements:
                        for cylce_edge in cycle_edges:
                            if cylce_edge in G_strategy.edges:
                                cycle_edges_to_remove.append(cylce_edge)
                                extention_type = 'partial'

                    # Add all inflowing nodes to cycle to superblock graph
                    all_internal_edges = []
                    for cycle_node in cycle_nodes:
                        neigh_nodes_4deg = list(G_local.neighbors(cycle_node)) 
                        for node_neighbouring_4degree in neigh_nodes_4deg:
                            inflow_edge = (cycle_node, node_neighbouring_4degree)

                            # Condition to count incoming edge
                            # REMOVED TO NOT CUTif (inflow_edge not in G_strategy.edges) and (inflow_edge[0] != inflow_edge[1]):
                            if (inflow_edge[0] != inflow_edge[1]):
                                G_strategy.add_edge(inflow_edge[0], inflow_edge[1])
                                for attribute, value in G_local.edges[inflow_edge].items():
                                    G_strategy.edges[inflow_edge][attribute] = value
                                G_strategy.edges[inflow_edge][label] = id_cnt
                                G_strategy.edges[inflow_edge]['inter_typ'] = extention_type
                                all_internal_edges.append(inflow_edge)

                    # Add cycles to superblock graph
                    for edge in cycle_edges:
                        if edge in cycle_edges_to_remove and remove_overlappin_elements: # edge was already considered
                            pass
                        else:
                            G_strategy.add_edge(edge[0], edge[1])
                            for attribute, value in G_local.edges[edge].items():
                                G_strategy.edges[edge][attribute] = value
                            G_strategy.edges[edge][label] = int(id_cnt)
                            G_strategy.edges[edge]['inter_typ'] = extention_type

                    # Store all edges in container
                    for edge in all_internal_edges:
                        if edge in container_blocks.keys():
                            container_blocks[edge].append(id_cnt)
                        else:
                            container_blocks[edge] = [id_cnt]
                    
                    id_cnt += 1

                else:
                    logging.info("Superblock densities not fulfilled {} {}".format(average_GFA_den, average_pop_den))

    # save all cycles used
    #gpd.GeoDataFrame([LineString(cycle) for cycle in used_cycles], columns=['geometry']).to_file(os.path.join("J:/Sven/_scrap/cycle.shp"))

    if G_strategy.number_of_edges() > 0:
        _, edges = hp_rw.nx_to_gdf(G_strategy)
    else:
        edges = gpd.GeoDataFrame()
        logging.info("No intervention found for superblock")

    return edges, G_strategy, container_blocks
