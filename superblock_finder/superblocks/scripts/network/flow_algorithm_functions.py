"""
Flow algorithm functions
TODO: Centrality is also quite a good measure
Note: 
"""
import os
import sys
import numpy as np
import networkx as nx
from networkx import connected_components
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path
import random
from scipy import spatial
from shapely.geometry import LineString

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)


def calc_flow_difference(G_base, G_flow, attribute, label):
    """
    Calculate difference of edge attribute of two graphs
    """
    
    nx.set_edge_attributes(G_base, 0, label)
    sum_diff_norm = 0
    diff_neg_norm = 0
    diff_pos_norm = 0
    tot_length = 0

    # Iterate flow network
    for edge in G_flow.edges:
        flow_edge = G_flow.edges[edge][attribute]
        flow_orig = G_base.edges[edge][attribute]
        length = G_flow.edges[edge]['geometry'].length
        diff = round(flow_edge - flow_orig, 5)
        G_base.edges[edge][label] = diff
        tot_length += G_flow.edges[edge]['geometry'].length

        sum_diff_norm += length * diff

        if diff < 0:  #  after interventino less flow
            diff_neg_norm += sum_diff_norm
        elif diff > 0:
            diff_pos_norm += sum_diff_norm
        else:
            pass

    # ----Normalized average difference (normalized with length)
    norm_diff = sum_diff_norm / tot_length

    # ---Compare savings versus gains
    norm_diff_pos_neg = diff_pos_norm - diff_neg_norm
    '''print("INfo differenr {}  {}".format(attribute, label))
    print("---")
    print(diff_pos_norm)
    print(diff_neg_norm)
    print(norm_diff_pos_neg)
    print("--")
    print(norm_diff)'''

    return G_base, norm_diff, norm_diff_pos_neg


def classify_flow_cat(
        G,
        label_out='flow_cat',
        label='flow_rel',
        input_type='graph',
        cats=[]
    ):
    """Create classification based on attribute
    """
    # Classification boundaries
    cat1 = cats[0]
    cat2 = cats[1]

    if input_type == 'graph':
        nx.set_edge_attributes(G, None, label_out)

        for edge in G.edges:
            flow_rel = G.edges[edge][label]
            if np.isnan(flow_rel):
                # If None, assign lowest category
                cat = 'low'
            else:
                if flow_rel <= cat1:
                    cat = 'low'
                elif flow_rel > cat1 and flow_rel <= cat2:
                    cat = 'medium'
                else:
                    cat = 'high'

            G.edges[edge][label_out] = cat

    if input_type == 'gdf':
        assert G.index.is_unique
        G[label_out] = None
        for index_g in G.index:
            flow_rel = G.loc[index_g][label]
            if np.isnan(flow_rel):                
                # If None, assign lowest category
                cat = 'low'
                #pass
            else:
                if flow_rel <= cat1:
                    cat = 'low'
                elif flow_rel > cat1 and flow_rel <= cat2:
                    cat = 'medium'
                else:
                    cat = 'high'

            G.at[index_g, label_out] = cat

    return G


def clean_network_and_assign_capacity(G, flow_factor=1, mode='tags.highway'):
    """Aasign base flow, i.e. lane capacity
    """
    # Convert to undicrected graph
    G_undirected_frozen = G.to_undirected(as_view=True) # as_view  --> Makes that edge attributes are kept
    G_undirected = nx.Graph(G_undirected_frozen)

    # Select largest network from all components
    components = list(connected_components(G_undirected))
    biggest_component_size = max(len(c) for c in components)
    problem_components = [c for c in components if len(c) != biggest_component_size] 
    for component in problem_components:
        for node in component:
            G_undirected.remove_node(node)

    # Define capacity of networks (use lane numbers), (use relative capacity based on line numbers)
     # Relative capacity according to lane number
    for edge in G_undirected.edges:
        if mode == 'random':
            lane_number = random.randint(1, 9)
        elif mode == 'constant':
            lane_number = 10
        elif mode == 'lanes':
            lane_number = G_undirected.edges[edge]['tags.lanes']
        elif mode == 'tags.highway':
            highway_tag = G_undirected.edges[edge]['tags.highway']

            lane_number = False
            if highway_tag in ['primary', 'primary_link', 'motorway', 'motorway_link']:
                lane_number = 4
            elif highway_tag in ['secondary', 'secondary_link']:
                lane_number = 3
            elif highway_tag in ['tertiary', 'tertiary_link']:
                lane_number = 2
            elif highway_tag in ['residential']:
                lane_number = 1
            else:
                lane_number = 1
        elif mode == 'combined_lane_and_type':

            lane_number = G_undirected.edges[edge]['tags.lanes']
            if not lane_number:
                if highway_tag in ['primary', 'primary_link', 'motorway', 'motorway_link']:
                    lane_number = 4
                elif highway_tag in ['secondary', 'secondary_link']:
                    lane_number = 3
                elif highway_tag in ['tertiary', 'tertiary_link']:
                    lane_number = 2
                elif highway_tag in ['residential']:
                    lane_number = 1
                else:
                    lane_number = 1   
        else:
            raise Exception("Define mode to assign capacity")

        if not lane_number:
            G_undirected.edges[edge]['capacity'] = 1 * flow_factor
        else:
            G_undirected.edges[edge]['capacity'] = int(lane_number) * flow_factor

    return G_undirected


def create_super_sinks_sources(
        G,
        bb,
        nr_help_pnts=20
    ):
    """TODO: IMprove: not only 4 dirrectinos, but stepwise degree
    """
    print("... create super sink and sources")
    G_undirected_frozen = G.to_undirected(as_view=True)
    G_undirected = nx.Graph(G_undirected_frozen)

    # -----------------------------
    # Crate super flow and super sink networks
    # [ ] For 4 directions
    #          s1  ---> assign closest network node
    #        / 
    #       /
    # SuperS-- s2  ---> assign closest network node
    #       \
    #        \  
    #          s3  ---> assign closest network node

    # -----------------------------
    kd_tree = spatial.KDTree([(node[0], node[1]) for node in G_undirected.nodes])
    dict_G_super = {}
    orientations = ['north', 'east', 'south', 'west' ]

    for orientation in orientations:
        G_super = nx.Graph()
        if orientation == 'west':
            edge_line = LineString(((bb.xmin, bb.ymax), (bb.xmin, bb.ymin)))
            super_pnt = ((bb.xmin - (bb.xmax - bb.xmin) / 2), (bb.ymin + (bb.ymax - bb.ymin) / 2))
        if orientation == 'east':
            edge_line = LineString(((bb.xmax, bb.ymax), (bb.xmax, bb.ymin)))
            super_pnt = ((bb.xmax + (bb.xmax - bb.xmin) / 2), (bb.ymin + (bb.ymax - bb.ymin) / 2))
        if orientation == 'north':
            edge_line = LineString(((bb.xmin, bb.ymax), (bb.xmax, bb.ymax)))
            super_pnt = ((bb.xmin + (bb.xmax - bb.xmin) / 2), (bb.ymax + (bb.ymax - bb.ymin) / 2))
        if orientation == 'south':
            edge_line = LineString(((bb.xmin, bb.ymin), (bb.xmax, bb.ymin)))
            super_pnt = ((bb.xmin + (bb.xmax - bb.xmin) / 2), (bb.ymin - (bb.ymax - bb.ymin) / 2))
        G_super.add_node(super_pnt) 


        # Prevent that direct infiniate link is generated (by remove outer 10 percent of spit pnts)
        limitnig_pnt_nrs = 5
        if nr_help_pnts < 2 * limitnig_pnt_nrs:
            raise Warning("Not eough points defined")

        # Create split pnts and create super sink/source network
        split_pnts = []
        #for i in range(1, nr_help_pnts):
        for i in range(limitnig_pnt_nrs, nr_help_pnts - limitnig_pnt_nrs):
            dist = i / nr_help_pnts
            pnt = edge_line.interpolate(dist, normalized=True)
            split_pnts.append((pnt.x, pnt.y))

        # --Assign closest network node with flow capacity to spit_pnt
        assigned_nodes_id = []
        split_pnts_unique = []  # List with all split point which can be assigned to unique network node

        for split_pnt in split_pnts:

            # Find closest network pnt
            nr_closest = 1
            closest_nodes = kd_tree.query(split_pnt, k=nr_closest)
            closest_network_node = list(G_undirected.nodes)[closest_nodes[1]]  # 0 is distance

            # True: Assign multiple splitnts to supernetwork
            crit_allow_multiple = True
            if closest_network_node not in assigned_nodes_id or crit_allow_multiple:

                # Get capacity of node by summing all capacity of edges
                neighbours_network_nodes = list(G_undirected.neighbors(closest_network_node))
                #capacity_network_edge = 0
                capacity_network_edge_list = []
                for neighbours_network_node in neighbours_network_nodes:
                    neigbor_edge = (closest_network_node, neighbours_network_node)
                    #capacity_network_edge += G_undirected.edges[neigbor_edge]['capacity']
                    capacity_network_edge_list.append(G_undirected.edges[neigbor_edge]['capacity'])

                # Take largest/smallest capacity
                #capacity_network_edge = max(capacity_network_edge_list)
                #capacity_network_edge = min(capacity_network_edge_list)
                #capacity_network_edge = 1
                # If not provided ==  infinite capacity

                # Add edges from network node to split point
                edge = (split_pnt, closest_network_node)
                G_super.add_edge(edge[0], edge[1])
                G_super.edges[edge]['geometry'] = LineString(edge)
                #G_super.edges[edge]['capacity'] = capacity_network_edge 
                #G_super.nodes[split_pnt]['capacity'] = capacity_network_edge  #NEW
                assigned_nodes_id.append(closest_network_node)
                split_pnts_unique.append(split_pnt)

        # Add edges from super sink to split points
        for split_pnt in split_pnts_unique:
            edge = (super_pnt, split_pnt)  # Note: because graph, direction does not matter
            G_super.add_edge(edge[0], edge[1])
            #G_super.edges[edge]['capacity'] = G_super.nodes[split_pnt]['capacity']  #NEW
            G_super.edges[edge]['geometry'] = LineString(edge)
        G_super.graph = G.graph

        dict_G_super[orientation] = {
            'super_graph': G_super,
            'super_pnt': super_pnt,
            "split_pnts": split_pnts_unique}

    # Compose supergraphs (sink and source) with graph
    G_undirected = nx.compose_all(
        (
            dict_G_super['north']['super_graph'],
            dict_G_super['west']['super_graph'],
            dict_G_super['south']['super_graph'],
            dict_G_super['east']['super_graph'],
            G_undirected
        ))

    # Calculate edge_betweenness_centrality
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html
    edges_centrality = nx.algorithms.edge_betweenness_centrality(G, weight='capacity')
    for edge, value in edges_centrality.items():
        G_undirected.edges[edge]['centrality'] = round(value, 5)

    # Add empty attribute for flow summarizing
    nx.set_edge_attributes(G_undirected, 0, "flow")
    nx.set_edge_attributes(G_undirected, 0, "sum_flow")

    return G_undirected, dict_G_super


def flow_emund(
        G_undirected,
        dict_G_super,
        max_road_cap,
        flow_factor=1,
        alg_type='edmund_karps'
    ):
    """Edmons-Karp algorithm (implementation of fulkerson)

    Note: Run for each combination of 
    """
    # ==================================================
    # Run from all 4 direction and sum_flow
    # ==================================================
    print("... Run Edmond-Karps algorithm in 4 directions")
    start_nodes = [
        'north',
        'east',
        'south',
        'west',
        ]
    end_nodes = [
        'south',
        'west',
        'north',
        'east',
        ]

    run_names = []
    cnt = -1
    for from_direction, to_direction in zip(start_nodes, end_nodes):

        start_node = dict_G_super[from_direction]['super_pnt']
        end_node = dict_G_super[to_direction]['super_pnt']

        cnt += 1
        run_name = "flow_{}".format(cnt)

        print("... flow algorithm iteration {}".format(cnt))
        print("Sink: {}   Source:   {}".format(from_direction, to_direction))

        # Only keep the fitting supernetwork and remove the super_graph which 
        # does not correspond with the right direction
        G_undirected_flow = G_undirected.copy()

        # Remove auxilary network not in correct direction
        for direction in dict_G_super.keys():
            if direction not in [from_direction, to_direction]:
                G_undirected_flow.remove_nodes_from(dict_G_super[direction]['split_pnts'])
                G_undirected_flow.remove_nodes_from(dict_G_super[direction]['super_pnt'])

        # ------------------------------
        # Flow algorithm
        # ------------------------------
        if alg_type == 'edmund_karps':
            # Flow algorithm. OUtput is residual netowrk () R.graph['flow_value']  #
            # A residual network graph indicates how much more flow is allowed in each edge in the network graph.
            try:
                R = edmonds_karp(G_undirected_flow, start_node, end_node, capacity="capacity")
                for edge in R.edges:
                    flow = abs(R.edges[edge]['flow'])   
                    G_undirected.edges[edge]['sum_flow'] += flow # Absolute value summed across different orientations
                    G_undirected.edges[edge][run_name] = flow   # Absolute value summed across different orientations
                run_names.append(run_name)
            except:
                print("WARNING: Could not calculate edmund karps for: {}  {}".format(from_direction, to_direction))
        if alg_type == 'shortest_augmenting_path':
            G_shotest_path = shortest_augmenting_path(G_undirected_flow, start_node, end_node, capacity="capacity")
            for edge in G_shotest_path.edges:
                flow = abs(G_shotest_path.edges[edge]['flow'])                        
                G_undirected.edges[edge]['sum_flow'] += flow # Absolute value summed across different orientations
                G_undirected.edges[edge][run_name] = flow   # Absolute value summed across different orientations
        if alg_type == 'maximum_flow':      
            flow_value, flow_dict = nx.maximum_flow(G_undirected_flow, start_node, end_node, capacity="capacity")
            for edge in G_undirected.edges:
                try:
                    flow = abs(flow_dict[edge[0]][edge[1]])                        
                    G_undirected.edges[edge]['sum_flow'] += flow
                    G_undirected.edges[edge][run_name] = flow
                except:
                    pass

        #nodes, edges = hp_rw.nx_to_gdf(G_undirected)
        #edges.to_file("C:/_scrap/base_flow_edmunds.shp")


    # Remove supergraph from original network
    directions = ['north', 'west', 'south', 'east']
    for direction in directions:
        G_undirected.remove_nodes_from(dict_G_super[direction]['split_pnts'])
        G_undirected.remove_nodes_from(dict_G_super[direction]['super_pnt'])

    nr_of_calculations = len(run_names)
    # ------------------------------------------------------------
    # Sum flow and calculate average
    # ---------------------------------------------
    nx.set_edge_attributes(G_undirected, 0, 'sum_flow')
    nx.set_edge_attributes(G_undirected, 0, 'av_flow')
    nx.set_edge_attributes(G_undirected, 0, 'calc_nrs')
    for edge in G_undirected.edges:
        G_undirected.edges[edge]['calc_nrs'] = nr_of_calculations
        sum_all_runs = 0
        for run_name in run_names:
            sum_all_runs += G_undirected.edges[edge][run_name]
        G_undirected.edges[edge]['sum_flow'] = sum_all_runs
        if sum_all_runs == 0:
            G_undirected.edges[edge]['av_flow'] = 0
        else:
            G_undirected.edges[edge]['av_flow'] = sum_all_runs / nr_of_calculations

    # ---------------------------------------------
    # Convert flow into percentage of maximum flow
    # (Normalize with maximum flow)
    # ---------------------------------------------
    max_flow = max_road_cap * nr_of_calculations * flow_factor
    nx.set_edge_attributes(G_undirected, 0, 'rel_flow')
    for edge in G_undirected.edges:
        sum_flow = G_undirected.edges[edge]['sum_flow']
        if sum_flow == 0:
            G_undirected.edges[edge]['rel_flow'] = 0
        else:
            G_undirected.edges[edge]['rel_flow'] = G_undirected.edges[edge]['sum_flow'] / max_flow

    print("Max Flow: {}".format(max_flow))

    return G_undirected
