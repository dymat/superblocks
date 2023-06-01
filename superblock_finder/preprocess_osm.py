"""
Download data from openstreetmap for superblock analysis

Note: Max 10,000 queries per day and download < 5 GB
https://wiki.openstreetmap.org/wiki/Overpass_API#:~:text=https%3A%2F%2Flz4.overpass%2Dapi.de%2Fapi%2Finterpreter&text=Any%20of%20the%20three%20servers,about%201%2C000%2C000%20requests%20per%20day.z
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_superblocks)

import geopandas as gpd
import networkx as nx

from shapely.geometry import Point
from sqlalchemy import create_engine

from superblocks.scripts.network import helper_osm as hp_osm
from superblocks.scripts.network import helper_network as hp_net
from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.population import population_extraction as pe


# WSG UTM ZONES: http://www.dmap.co.uk/utmworld.htm
# 4326: WSG 84   
# 32633: UTM N

# Note: If downloaded by OSM, use UTM projections. Otherwise meter calulations don't work
# UTM Switzerland epsg:32632
crs_pop_fb = 4326
crs_bb = 4326
crs_overpass = 4326

path_temp = "/data/tmp"
path_pop_data = "/deu_pd_2020_1km.tif"  # Download data as outlined from data source in publication
try:
    os.makedirs(path_temp)
except:
    pass

# Window selection
length_in_m = 15000  # [m]
radius_pop_density = 100  # [m]
radius_GFA_density = 100 # [m]

case_studies = [
    #'deutschland',
    #'atlanta',
    #'bankok',        
    #'barcelona',     
    #'bremen',
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
    #'hamburg',
    #'munchen'
]


postgis_connection = create_engine(f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
                     f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}" \
                     f"@{os.getenv('POSTGRES_HOST', 'localhost')}:5432/{os.getenv('POSTGRES_DATABASE', 'postgres')}")


for city in case_studies:
    city_metadata = hp_rw.city_metadata(city, path_pop_data=path_pop_data)
    to_crs_meter = city_metadata['crs']
    path_raw_pop = city_metadata['path_raw_pop']
    centroid_tuple = city_metadata['centroid_tuple']
    print("=== city: {}".format(city))


    # Download first bounding box shp (as WSG84 crs)
    centroid = Point(centroid_tuple)
    centroid = gpd.GeoDataFrame([centroid], columns=['geometry'], crs=crs_bb)

    centroid_to_crs_meter = centroid.to_crs("epsg:{}".format(to_crs_meter))
    bb = hp_osm.BB(
        ymax=centroid_to_crs_meter.geometry.y[0] + length_in_m / 2,
        ymin=centroid_to_crs_meter.geometry.y[0] - length_in_m / 2,
        xmax=centroid_to_crs_meter.geometry.x[0] + length_in_m / 2,
        xmin=centroid_to_crs_meter.geometry.x[0] - length_in_m / 2)

    bb_gdf = bb.as_gdf(crs_orig=to_crs_meter)

    # bb to postgis
    bb_gdf.to_postgis("bbox", postgis_connection, if_exists="replace")

    bb_osm = bb_gdf.to_crs("epsg:{}".format(crs_bb))
    bb = hp_osm.BB(
        ymax=bb_osm.geometry.bounds.maxy[0],
        ymin=bb_osm.geometry.bounds.miny[0],
        xmax=bb_osm.geometry.bounds.maxx[0],
        xmin=bb_osm.geometry.bounds.minx[0])

    # ==============================================================================
    # The following overpass turbo commands have been implemented
    # ==============================================================================
    print("... extracting general OSM overvpass")

    bb_geom = list(bb_gdf.geometry)[0]

    # Check if streets were downloaded and thus all geomeetries:

    # Download buildings
    osm_buildings_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='buildings')
    if osm_buildings_gdf.shape[0] > 0:
        osm_buildings_gdf = hp_net.clip_outer_polygons(osm_buildings_gdf, bb_geom)

        # Remove invalid polygons
        osm_buildings_gdf = hp_osm.remove_faulty_polygons(osm_buildings_gdf)

        # bb to postgis
        osm_buildings_gdf.to_postgis("buildings", postgis_connection, if_exists="replace")

    # Download water
    osm_water_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='water')
    if osm_water_gdf.shape[0] > 0:
        osm_water_gdf = hp_net.clip_outer_polygons(osm_water_gdf, bb_geom)
        if osm_water_gdf.shape[0] > 0:
            osm_water_gdf = hp_net.gdf_multilinestring_to_linestring(osm_water_gdf)
            osm_water_gdf.to_postgis("water", postgis_connection, if_exists="replace")

    # Download bus
    osm_bus_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bus')
    if osm_bus_gdf.shape[0] > 0:
        osm_bus_gdf = hp_net.clip_outer_polygons(osm_bus_gdf, bb_geom)
        if osm_bus_gdf.shape[0] > 0:
            osm_bus_gdf = hp_net.gdf_multilinestring_to_linestring(osm_bus_gdf)
            osm_bus_gdf.to_postgis("bus", postgis_connection, if_exists="replace")

    # Download bridges
    osm_bridges_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bridges')
    if osm_bridges_gdf.shape[0] > 0:
        osm_bridges_gdf = hp_net.clip_outer_polygons(osm_bridges_gdf, bb_geom)
        if osm_bridges_gdf.shape[0] > 0:
            osm_bridges_gdf.to_postgis("bridges", postgis_connection, if_exists="replace")

    # Download land use
    osm_landuse_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='landuse')
    if osm_landuse_gdf.shape[0] > 0:
        osm_landuse_gdf = hp_net.clip_outer_polygons(osm_landuse_gdf, bb_geom)
        if osm_landuse_gdf.shape[0] > 0:
            osm_landuse_gdf.to_postgis("landuse", postgis_connection, if_exists="replace")

    # Download tram
    osm_tram_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='tram')
    if osm_tram_gdf.shape[0] > 0:
        osm_tram_gdf = hp_net.gdf_multilinestring_to_linestring(osm_tram_gdf)
        osm_tram_gdf = hp_net.clip_outer_polygons(osm_tram_gdf, bb_geom)
        if osm_tram_gdf.shape[0] > 0:
            osm_tram_gdf.to_postgis("tram", postgis_connection, if_exists="replace")

    # Download bus
    osm_trolleybus_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='trolleybus')
    if osm_trolleybus_gdf.shape[0] > 0:
        osm_trolleybus_gdf = hp_net.gdf_multilinestring_to_linestring(osm_trolleybus_gdf)
        osm_trolleybus_gdf = hp_net.clip_outer_polygons(osm_trolleybus_gdf, bb_geom)
        if osm_trolleybus_gdf.shape[0] > 0:
            osm_trolleybus_gdf.to_postgis("trolleybus", postgis_connection, if_exists="replace")

    # Download streets
    osm_streets_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='streets')

    if osm_streets_gdf.shape[0] > 0:
        osm_streets_gdf = hp_net.gdf_multilinestring_to_linestring(osm_streets_gdf)  # Multiline gdf to singline gdf
        osm_streets_gdf = hp_net.clip_outer_polygons(osm_streets_gdf, bb_geom)
        osm_streets_gdf = hp_net.remove_all_intersections(osm_streets_gdf) # remove all intersections (lines which overlap)
        osm_streets_gdf = hp_net.remove_rings(osm_streets_gdf)
        G_streets = hp_rw.gdf_to_nx(osm_streets_gdf)
        del osm_streets_gdf
        G_streets = hp_net.simplify_network(
            G_streets, crit_big_roads=False, crit_bus_is_big_street=False)

    print("downloaded simple street: {}".format(city))
    print("... downloaded data {}".format(city))

    # ==============================================================================
    # Assign  attributes from one graph to another graph
    # ==============================================================================

    # Remove footway (new)
    G_streets = hp_net.remove_edge_by_attribute(G_streets, attribute='tags.highway', value="footway")
    G_streets = hp_net.G_multilinestring_to_linestring(G_streets, single_segments=True)

    # Criteria which influcence how graphs are spatially merged
    buffer_dist = 10        # [m]
    min_edge_distance = 10  # [m]
    p_min_intersection = 80 # [m] minimum edge percentage which needs to be intersected

    G_streets = hp_net.simplify_network(
        G_streets, crit_big_roads=False, crit_bus_is_big_street=False)

    nx.set_edge_attributes(G_streets, 0, 'bus')
    nx.set_edge_attributes(G_streets, 0, 'tram')
    nx.set_edge_attributes(G_streets, 0, 'trolleybus')

    if osm_bus_gdf.shape[0] > 0: 
        # TODO: get bus from postgis
        G_bus = hp_rw.gdf_to_nx(osm_bus_gdf)
        G_streets = hp_net.check_if_paralell_lines(
            G_bus, G_streets, crit_buffer=buffer_dist,
            min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='bus')
    del osm_bus_gdf

    if osm_tram_gdf.shape[0] > 0: 
        # TODO: get tram from postgis
        G_tram = hp_rw.gdf_to_nx(osm_tram_gdf)
        G_streets = hp_net.check_if_paralell_lines(
            G_tram, G_streets, crit_buffer=buffer_dist,
            min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='tram')
    del osm_tram_gdf


    if osm_trolleybus_gdf.shape[0] > 0:
        # TODO: get trolleybus from postgis
        G_trolleybus = hp_rw.gdf_to_nx(osm_trolleybus_gdf)
        G_streets = hp_net.check_if_paralell_lines(
            G_trolleybus, G_streets, crit_buffer=buffer_dist,
            min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='trolleybus')
    del osm_trolleybus_gdf


    print("assign_attributes_to_graph finished")

    # ==============================================================================
    # ---- Assign population density to network
    # ==============================================================================
    print("calculate_pop_density start")

    # Load facebook popluation data
    # TODO: get street edges with attributes from postgis
    
    _, gdf_street = hp_rw.nx_to_gdf(G_streets)

    # Get bounding box
    bb = hp_osm.BB(
        ymax=max(gdf_street.geometry.bounds.maxy),
        ymin=min(gdf_street.geometry.bounds.miny),
        xmax=max(gdf_street.geometry.bounds.maxx),
        xmin=min(gdf_street.geometry.bounds.minx))
    bb_pop = bb.as_gdf(crs_orig=int(gdf_street.crs.srs.split(":")[1]))

    if city_metadata['data_type'] == 'GeoTif':
        pop_pnts = pe.get_fb_pop_data_tif(bb_pop, crs_pop_fb, path_temp=path_temp, path_raw=path_raw_pop, label='population')
    elif city_metadata['data_type'] == 'csv':
        pop_pnts = pe.get_fb_pop_data(bb_pop, crs_pop_fb, path_raw=path_raw_pop, label='population')
    else:
        raise Exception("Wrong format type")
    pop_pnts.to_postgis("fb_pop", postgis_connection, if_exists="replace")

    # ----Calculate population density
    G_streets = hp_net.calc_edge_and_node_pop_density(
        G_streets,
        pop_pnts,
        radius=radius_pop_density,
        attribute_pop='population',
        label='pop_den')

    # ----Calculate GFA density based on osm buildings
    G_streets = hp_net.calc_GFA_density(G_streets, osm_buildings_gdf, radius=radius_GFA_density, label='GFA_den')

    print("writing out shp")
    nodes, edges = hp_rw.nx_to_gdf(G_streets)

    nodes.to_postgis("street_network_nodes_with_attributes_pop_density", postgis_connection, if_exists="replace")
    edges.to_postgis("street_network_edges_with_attributes_pop_density", postgis_connection, if_exists="replace")

print("-----finished script-----")
