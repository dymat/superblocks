"""This is closely linked to momepy.nx_to_gdf
but customized
"""
import os
import json
import networkx as nx
import geopandas as gpd
import logging
from shapely.geometry import Point, mapping, shape

def city_labels():
    """Labels
    """
    label_dict = {
        'atlanta': 'Atlanta',
        'bankok': 'Bangkok',
        'barcelona': 'Barcelona',
        'berlin': 'Berlin',
        'budapest': 'Budapest',
        'cairo': 'Cairo',
        'hong_kong': 'Hong Kong',
        'lagos': 'Lagos',
        'london': 'London',
        'madrid': 'Madrid',
        'melbourne': 'Melbourne',
        'mexico_city': 'Mexico City',
        'paris': 'Paris',
        'rome': 'Rome',
        'sydney': 'Sydney',
        'tokyo': 'Tokyo',
        'warsaw': 'Warsaw',
        'zurich': 'Zürich',
        'frankfurt': 'Frankfurt',
        'freiburg': 'Freiburg',
        'hamburg': 'Hamburg',
        'munchen': 'München',
        }

    return label_dict

def city_labels_ch():
    """Labels
    """
    label_dict = {
        'Luzern': 'Lucerne',
        'Genève': 'Geneva',
        }

    return label_dict

def city_metadata(city, path_pop_data="C:/DATA/pop_fb"):
    """UTM ZONES: http://www.dmap.co.uk/utmworld.htm

    Note: Centroid need to be provided from 4326
    """
    metadata = {
        'deutschland': {
            'data_type': 'GeoTif',
            'centroid_tuple': (9.772966975048563, 52.37886144060916),
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'zurich': {
            'data_type': 'GeoTif',
            'centroid_tuple': (8.536838, 47.372925),
            'path_raw_pop': os.path.join(path_pop_data, "che_general_2020.tif"),
            'crs': 32632},
        'barcelona': {
            'data_type': 'GeoTif',
            'centroid_tuple': (2.155307, 41.388285),
            'path_raw_pop': os.path.join(path_pop_data, "population_esp_2019-07-01.tif"),
            'crs': 32630},
        'aarau': {
            'data_type': 'csv',
            'centroid_tuple': (8.045030, 47.391035),
            'path_raw_pop': os.path.join(path_pop_data, "population_che_2019-07-01.csv"),
            'crs': 32632},
        'paris': {
            'data_type': 'GeoTif',
            'centroid_tuple': (2.343122, 48.871521),
            'path_raw_pop': os.path.join(path_pop_data, "population_fra_2019-07-01.tif"),
            'crs': 32631},
        'atlanta': {
            'data_type': 'GeoTif',
            'centroid_tuple': (-84.393214, 33.759360),
            'path_raw_pop': os.path.join(path_pop_data, "population_usa28_-90_2019-07-01.tif"),
            'crs': 32616},
        'bankok': {
            'data_type': 'GeoTif',
            'centroid_tuple': (100.493726, 13.749962),
            'path_raw_pop': os.path.join(path_pop_data, "thailand_2020.tif"),
            'crs': 32647},
        'bejing': { #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},
        'berlin': {
            'data_type': 'GeoTif',
            'centroid_tuple': (13.4122611, 52.5225496),  
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'frankfurt': {
            'data_type': 'GeoTif',
            'centroid_tuple': (8.67000401, 50.10835373),  
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'freiburg': {
            'data_type': 'GeoTif',
            'centroid_tuple': (7.84604697, 47.99544959),  
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'hamburg': {
            'data_type': 'GeoTif',
            'centroid_tuple': (10.0069566, 53.5530345),  
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'munchen': {
            'data_type': 'GeoTif',
            'centroid_tuple': (11.576216, 48.135332),  
            'path_raw_pop': os.path.join(path_pop_data),
            'crs': 32632},
        'budapest': {
            'data_type': 'GeoTif',
            'centroid_tuple': (19.05013, 47.50533),
            'path_raw_pop': os.path.join(path_pop_data, "population_hun_2019-07-01.tif"),
            'crs': 32634},
        'cairo': {
            'data_type': 'GeoTif',
            'centroid_tuple': (31.243793, 30.050229),
            'path_raw_pop': os.path.join(path_pop_data, "population_egy_2018-10-01.tif"),
            'crs': 32634},
        'dubai': { # does not make sense
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},
        'hong_kong': {
            'data_type': 'GeoTif',
            'centroid_tuple': (114.174124, 22.319044),
            'path_raw_pop': os.path.join(path_pop_data, "population_hkg_2018-10-01.tif"),
            'crs': 32649},
        'lagos': {
            'data_type': 'GeoTif',
            'centroid_tuple': (3.352456, 6.518423),
            'path_raw_pop': os.path.join(path_pop_data, "population_nga_2018-10-01.tif"),
            'crs': 23031},  
        'london': {
            'data_type': 'GeoTif',
            'centroid_tuple': (-0.089675, 51.512577),
            'path_raw_pop': os.path.join(path_pop_data, "population_gbr_2019-07-01.tif"),
            'crs': 32630},  
        'madrid': {
            'data_type': 'GeoTif',
            'centroid_tuple': (-3.7036030, 40.4168092),
            'path_raw_pop': os.path.join(path_pop_data, "population_esp_2019-07-01.tif"),
            'crs': 32630},
        'melbourne': {
            'data_type': 'GeoTif',
            'centroid_tuple': (144.966343, -37.811835),
            'path_raw_pop': os.path.join(path_pop_data, "population_aus_southeast_2018-10-01.tif"),
            'crs': 32755},  
        'mexico_city': {
            'data_type': 'GeoTif',
            'centroid_tuple': (-99.1330489, 19.4324909),
            'path_raw_pop': os.path.join(path_pop_data, "population_mex_2018-10-01.tif"),
            'crs': 32614},  
        'moscow': {  #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs':  None},  
        'mumbai': {  #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},  
        'rio_de_janeiro': { #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},
        'rome': {
            'data_type': 'GeoTif',
            'centroid_tuple': (12.495526, 41.901780),
            'path_raw_pop': os.path.join(path_pop_data, "population_ita_2019-07-01.tif"),
            'crs': 32632},
        'seoul': {
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},        
        'sydney': {
            'data_type': 'GeoTif',
            'centroid_tuple': (151.204664, -33.883037),
            'path_raw_pop': os.path.join(path_pop_data, "population_aus_southeast_2018-10-01.tif"),
            'crs': 32755},
        'tehran': {  #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,
            'crs': None},  
        'tokyo': {
            'data_type': 'GeoTif',
            'centroid_tuple': (139.770194, 35.683808),
            'path_raw_pop': os.path.join(path_pop_data, "jpn_population_2020.tif"),
            'crs': 32653},
        'toronto': {  #no data
            'data_type': 'csv',
            'centroid_tuple': None,
            'path_raw_pop': None,  # NOD DATA AVAILABLE
            'crs':  None},
        'warsaw': {
            'data_type': 'GeoTif',
            'centroid_tuple': (21.011328, 52.246989),
            'path_raw_pop': os.path.join(path_pop_data, "population_pol_2019-07-01.tif"),
            'crs': 32634},
    }

    city_meta = metadata[city]

    return city_meta


def get_largest_polygon(list_with_polygons):
    """Select largest polygon in a list
    """
    area = 0
    for geom_obj in list_with_polygons:
        if geom_obj.area > area:
            largest_polygon = geom_obj
            area = geom_obj.area

    return largest_polygon


def get_files_in_folder(path_folder, ending=False):
    """
    """
    if ending:
        all_files_raw = os.listdir(path_folder)
        all_files = []
        for file_name in all_files_raw:
            if file_name.endswith(ending):
                all_files.append(file_name)
    else:
        all_files = os.listdir(path_folder)

    return all_files


def cm2inch(*tupl):
    """Convert input cm to inches (width, hight)
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def create_folder(path_folder, name_subfolder=None):
    """Creates folder or subfolder

    Arguments
    ----------
    path : str
        Path to folder
    folder_name : str, default=None
        Name of subfolder to create
    """
    if not name_subfolder:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
    else:
        path_result_subolder = os.path.join(path_folder, name_subfolder)
        if not os.path.exists(path_result_subolder):
            os.makedirs(path_result_subolder)


def gdf_to_nx(
        gdf_network,
        type='Graph',
        directional=False,
        tag_label='tags.one'
    ):
    """gdf to nx.DiGraph

    Geometry attribute but is stored on edge as 'geometry'

    """
    assert 'geometry' in gdf_network.columns.tolist()
    assert gdf_network.index.is_unique
    assert gdf_network.crs.srs  # note None

    # Add jason and wkt geometry
    #gdf_network['wkt'] = [i.wkt for i in gdf_network.geometry]
    #gdf_network.loc['Json'] = [json.dumps(mapping(i)) for i in gdf_network.geometry] 

    fields = list(gdf_network.columns)

    if type == 'MultiGraph':
        G = nx.MultiGraph()
    elif type == 'DiGraph':
        G = nx.DiGraph()
    elif type == 'Graph':
        G = nx.Graph()

    G.graph['crs'] = gdf_network.crs

    for index, row in gdf_network.explode().iterrows():
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]
        geometry = row.geometry

        data = [row[f] for f in fields]
        attributes = dict(zip(fields, data))
        if directional:
            if row[tag_label] == 'yes':
                G.add_edge(first, last, **attributes)
                G.edges[(first, last)]['geometry'] = geometry
            else:
                G.add_edge(first, last, **attributes)
                G.add_edge(last, first, **attributes)
                G.edges[(first, last)]['geometry'] = geometry
                G.edges[(last, first)]['geometry'] = geometry
        else:
            G.add_edge(first, last, **attributes)
            #G.add_edge(last, first, **attributes)
            G.edges[(first, last)]['geometry'] = geometry
            #G.edges[(last, first)]['geometry'] = geometry  #Note: MAYBE WRONG DIRECTION

    return G


def _points_to_gdf(net):
    """
    Generate point gdf from nodes.
    Helper for nx_to_gdf.
    """
    node_xy, node_data = zip(*net.nodes(data=True))
    if isinstance(node_xy[0], int) and "x" in node_data[0].keys():
        geometry = [Point(data["x"], data["y"]) for data in node_data]  # osmnx graph
    else:
        geometry = [Point(*p) for p in node_xy]

    gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=geometry)
    gdf_nodes.crs = net.graph["crs"]

    return gdf_nodes


def _lines_to_gdf(net, nodeID):
    """
    Generate linestring gdf from edges.
    Helper for nx_to_gdf.
    """
    assert 'geometry' in net.edges[list(net.edges)[0]], "No geometry defined"

    starts, ends, edge_data = zip(*net.edges(data=True))

    node_start = []
    node_end = []
    for s in starts:
        node_start.append(net.nodes[s][nodeID])
    for e in ends:
        node_end.append(net.nodes[e][nodeID])

    # Edge geometry from colum
    gdf_edges = gpd.GeoDataFrame(list(edge_data))
    gdf_edges.crs = net.graph["crs"]

    gdf_edges["node_start"] = node_start
    gdf_edges["node_end"] = node_end

    return gdf_edges

def nx_to_gdf(net, nodeID="nodeID"):
    """G to gdf
    """
    assert 'crs' in net.graph.keys()

    cnt = 0
    for n in net:
        net.nodes[n][nodeID] = cnt
        cnt += 1

    # Create nodes
    gdf_nodes = _points_to_gdf(net)

    # Create edges
    gdf_edges = _lines_to_gdf(net, nodeID)

    return gdf_nodes, gdf_edges


def set_up_logger(path_log_file, mode='INFO'):
    """Create logger
    Arguments
    --------
    path_log_file : str
        Path to logger file
    Info
    -----
    The logging level can be changed depending on mode
    Note
    ----
    logger.debug('debug message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')
    """
    # Create logging file if not existing
    if not os.path.isfile(path_log_file):
        open(path_log_file, 'w').close()

    # Set logger level
    if mode == 'INFO':
        logging.basicConfig(
            filename=path_log_file,
            filemode='w',  #'a, w'
            level=logging.INFO, #INFO, DEBUG, ERROR, CRITICAL
            format=('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    else:
        raise Exception("Extend function")
    # Necessary to add loggers in visual studio console
    ##logging.getLogger().addHandler(logging.StreamHandler())

    # Turn on/off logger
    #logging.disable = False
    #logging.disable(logging.CRITICAL)


def export_legend(legend, filename="legend.png"):
    """Export legend as seperate file
    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

