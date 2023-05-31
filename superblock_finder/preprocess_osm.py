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
import time
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
path_out = "/data/cities"
path_pop_data = "/deu_pd_2020_1km.tif"  # Download data as outlined from data source in publication
write_anyway = False

try:
    os.makedirs(path_out)
except:
    pass

try:
    os.makedirs(path_temp)
except:
    pass

hp_rw.create_folder(path_out)

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
    path_out_city = os.path.join(path_out, str(city))
    city_metadata = hp_rw.city_metadata(city, path_pop_data=path_pop_data)
    to_crs_meter = city_metadata['crs']
    path_raw_pop = city_metadata['path_raw_pop']
    centroid_tuple = city_metadata['centroid_tuple']
    print("=== city: {}".format(city))

    hp_rw.create_folder(path_out_city)

    # Download first bounding box shp (as WSG84 crs)
    centroid = Point(centroid_tuple)
    centroid = gpd.GeoDataFrame([centroid], columns=['geometry'], crs=crs_bb)
    #centroid.to_file(os.path.join(path_out_city, "centroid.shp"))

    centroid_to_crs_meter = centroid.to_crs("epsg:{}".format(to_crs_meter))
    #centroid.to_file(os.path.join(path_out_city, "centroid_{}.shp".format(to_crs_meter)))
    bb = hp_osm.BB(
        ymax=centroid_to_crs_meter.geometry.y[0] + length_in_m / 2,
        ymin=centroid_to_crs_meter.geometry.y[0] - length_in_m / 2,
        xmax=centroid_to_crs_meter.geometry.x[0] + length_in_m / 2,
        xmin=centroid_to_crs_meter.geometry.x[0] - length_in_m / 2)

    bb_gdf = bb.as_gdf(crs_orig=to_crs_meter)

    # bb to postgis
    bb_gdf.to_postgis("bbox", postgis_connection, if_exists="replace")

    #bb_gdf.to_file(os.path.join(path_out_city, "extent.shp"))
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
    path_landuse = os.path.join(path_out_city, "osm_landuse.shp")
    path_buildings = os.path.join(path_out_city, "osm_buildings.shp")
    path_tram = os.path.join(path_out_city, "tram.shp")
    path_bus = os.path.join(path_out_city, "bus.shp")
    path_trolleybus = os.path.join(path_out_city, "trolleybus.shp")
    path_bridges = os.path.join(path_out_city, "bridges.shp")
    path_water = os.path.join(path_out_city, "water.shp")
    path_streets_nodes = os.path.join(path_out_city, 'street_network_nodes.shp')
    path_streets_edges = os.path.join(path_out_city, 'street_network_edges.shp')
    path_complete_streets_nodes = os.path.join(path_out_city, 'street_complete_network_nodes.shp')
    path_complete_streets_edges = os.path.join(path_out_city, 'street_complete_network_edges.shp')

    bb_geom = list(bb_gdf.geometry)[0]

    # Check if streets were downloaded and thus all geomeetries:

    # Download buildings
    osm_buildings_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='buildings')
    if osm_buildings_gdf.shape[0] > 0:
        osm_buildings_gdf = hp_net.clip_outer_polygons(osm_buildings_gdf, bb_geom)

        # Remove invalid polygons
        osm_buildings_gdf = hp_osm.remove_faulty_polygons(osm_buildings_gdf)

        # TODO: buildings to postgis
        # bb to postgis
        osm_buildings_gdf.to_postgis("buildings", postgis_connection, if_exists="replace")
        #osm_gdf.to_file(path_buildings)

    # Download water
    osm_water_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='water')
    if osm_water_gdf.shape[0] > 0:
        osm_water_gdf = hp_net.clip_outer_polygons(osm_water_gdf, bb_geom)
        if osm_water_gdf.shape[0] > 0:
            osm_water_gdf = hp_net.gdf_multilinestring_to_linestring(osm_water_gdf)

            # TODO: water to postgis
            osm_water_gdf.to_postgis("water", postgis_connection, if_exists="replace")
            #osm_water_gdf.to_file(path_water)

    # Download bus
    osm_bus_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bus')
    if osm_bus_gdf.shape[0] > 0:
        osm_bus_gdf = hp_net.clip_outer_polygons(osm_bus_gdf, bb_geom)
        if osm_bus_gdf.shape[0] > 0:
            osm_bus_gdf = hp_net.gdf_multilinestring_to_linestring(osm_bus_gdf)

            # TODO: bus to postgis
            osm_bus_gdf.to_postgis("bus", postgis_connection, if_exists="replace")
            #osm_bus_gdf.to_file(path_bus)

    # Download bridges
    osm_bridges_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bridges')
    if osm_bridges_gdf.shape[0] > 0:
        osm_bridges_gdf = hp_net.clip_outer_polygons(osm_bridges_gdf, bb_geom)
        if osm_bridges_gdf.shape[0] > 0:

            # TODO: bridges to postgis
            osm_bridges_gdf.to_postgis("bridges", postgis_connection, if_exists="replace")
            #osm_bridges_gdf.to_file(path_bridges)

    # Download land use
    osm_landuse_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='landuse')
    if osm_landuse_gdf.shape[0] > 0:
        osm_landuse_gdf = hp_net.clip_outer_polygons(osm_landuse_gdf, bb_geom)
        if osm_landuse_gdf.shape[0] > 0:

            # TODO: landuse to postgis
            osm_landuse_gdf.to_postgis("landuse", postgis_connection, if_exists="replace")
            #osm_landuse_gdf.to_file(path_landuse)

    # Download tram
    osm_tram_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='tram')
    if osm_tram_gdf.shape[0] > 0:
        osm_tram_gdf = hp_net.gdf_multilinestring_to_linestring(osm_tram_gdf)
        osm_tram_gdf = hp_net.clip_outer_polygons(osm_tram_gdf, bb_geom)
        if osm_tram_gdf.shape[0] > 0:

            # TODO: tram to postgis
            osm_tram_gdf.to_postgis("tram", postgis_connection, if_exists="replace")
            #osm_tram_gdf.to_file(path_tram)

    # Download bus
    osm_trolleybus_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='trolleybus')
    if osm_trolleybus_gdf.shape[0] > 0:
        osm_trolleybus_gdf = hp_net.gdf_multilinestring_to_linestring(osm_trolleybus_gdf)
        osm_trolleybus_gdf = hp_net.clip_outer_polygons(osm_trolleybus_gdf, bb_geom)
        if osm_trolleybus_gdf.shape[0] > 0:

            # TODO: trolleybus to postgis
            osm_trolleybus_gdf.to_postgis("trolleybus", postgis_connection, if_exists="replace")
            #osm_trolleybus_gdf.to_file(path_trolleybus)

    # Download streets
    osm_streets_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='streets')

    if osm_streets_gdf.shape[0] > 0:
        osm_streets_gdf = hp_net.gdf_multilinestring_to_linestring(osm_streets_gdf)  # Multiline gdf to singline gdf
        osm_streets_gdf = hp_net.clip_outer_polygons(osm_streets_gdf, bb_geom)
        osm_streets_gdf = hp_net.remove_all_intersections(osm_streets_gdf) # remove all intersections (lines which overlap)
        osm_streets_gdf = hp_net.remove_rings(osm_streets_gdf)
        G = hp_rw.gdf_to_nx(osm_streets_gdf)
        G_simple = hp_net.simplify_network(
            G, crit_big_roads=False, crit_bus_is_big_street=False)

        print("main func", G_simple)

        # TODO: street network (Nodes + Edges to postgis)
        nodes, edges = hp_rw.nx_to_gdf(G_simple)
        #nodes.to_file(path_streets_nodes)
        #edges.to_file(path_streets_edges)

        print("Streets", edges.shape)

        nodes.to_postgis("street_network_nodes", postgis_connection, if_exists="replace")
        edges.to_postgis("street_network_edges", postgis_connection, if_exists="replace")

    print("downloaded simple street: {}".format(city))

    # Download complete street network
    '''if not os.path.exists(path_complete_streets_edges) or write_anyway:
        time.sleep(sleep_time)
        osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='all_streets')

        if osm_gdf.shape[0] > 0:
            #osm_gdf.to_file(os.path.join(path_out_city, 'street_raw.shp'))
            #osm_gdf = gpd.read_file(os.path.join(path_out_city, 'street_raw.shp'))
            osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)  # Multiline gdf to singline gdf

            # Clip
            osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)

            # remove all intersections (lines which overlopa)
            osm_gdf = hp_net.remove_all_intersections(osm_gdf)

            # Remove rings
            osm_gdf = hp_net.remove_rings(osm_gdf)

            G = hp_rw.gdf_to_nx(osm_gdf)
            G_simple = hp_net.simplify_network(G, crit_big_roads=False, crit_bus_is_big_street=crit_bus_is_big_street)

            nodes, edges = hp_rw.nx_to_gdf(G_simple)
            nodes.to_file(path_complete_streets_nodes)
            edges.to_file(path_complete_streets_edges)'''

    print("... downloaded data {}".format(city))

    #if not os.path.exists(os.path.join(path_out_city, 'street_network_edges_with_attributes.shp')) or write_anyway:
    # ==============================================================================
    # Assign  attributes from one graph to another graph
    # ==============================================================================
    #path_tram = os.path.join(path_out_city, "tram.shp")
    #path_bus = os.path.join(path_out_city, "bus.shp")
    #path_street = os.path.join(path_out_city, "street_network_edges.shp")
    #path_trolleybus = os.path.join(path_out_city, "trolleybus.shp")


    #gdf_tram = gpd.read_postgis("SELECT * FROM tram", postgis_connection)
    if osm_tram_gdf.shape[0] > 0: # os.path.exists(path_tram):
        # TODO: get tram from postgis
        #gdf_tram = gpd.read_file(path_tram)
        G_tram = hp_rw.gdf_to_nx(osm_tram_gdf)

    #gdf_bus = gpd.read_postgis("SELECT * FROM bus", postgis_connection)
    if osm_bus_gdf.shape[0] > 0: #os.path.exists(path_bus):
        # TODO: get bus from postgis
        #gdf_bus = gpd.read_file(path_bus)
        G_bus = hp_rw.gdf_to_nx(osm_bus_gdf)

    #gdf_trolleybus = gpd.read_postgis("SELECT * FROM trolleybus", postgis_connection)
    if osm_trolleybus_gdf.shape[0] > 0:
        # TODO: get trolleybus from postgis
        #gdf_trolleybus = gpd.read_file(path_trolleybus)
        G_trolleybus = hp_rw.gdf_to_nx(osm_trolleybus_gdf)

    #gdf_streets = gpd.read_postgis("SELECT * FROM street_network_edges", postgis_connection)
    gdf_streets = edges
    if gdf_streets.shape[0] > 0:
        # TODO: get street (=edges) from postgis
        #gdf_streets = gpd.read_file(path_street)
        if G_simple:
            G_streets = G_simple
        else:
            G_streets = hp_rw.gdf_to_nx(gdf_streets)

        # Remove footway (new)
        G_streets = hp_net.remove_edge_by_attribute(G_streets, attribute='tags.highway', value="footway")
        G_streets = hp_net.G_multilinestring_to_linestring(G_streets, single_segments=True)

        # Criteria which influcence how graphs are spatially merged
        buffer_dist = 10        # [m]
        min_edge_distance = 10  # [m]
        p_min_intersection = 80 # [m] minimum edge percentage which needs to be intersected

        G_streets = hp_net.simplify_network(
            G_streets, crit_big_roads=False, crit_bus_is_big_street=False)

        nx.set_edge_attributes(G_streets, 0, 'tram')
        nx.set_edge_attributes(G_streets, 0, 'bus')
        nx.set_edge_attributes(G_streets, 0, 'trolleybus')

        if osm_bus_gdf.shape[0] > 0:
            # TODO: get bus, tram and trollybus from postgis
            G_streets = hp_net.check_if_paralell_lines(
                G_bus, G_streets, crit_buffer=buffer_dist,
                min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='bus')

        if osm_tram_gdf.shape[0] > 0:
            G_streets = hp_net.check_if_paralell_lines(
                G_tram, G_streets, crit_buffer=buffer_dist,
                min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='tram')

        if osm_trolleybus_gdf.shape[0] > 0:
            G_streets = hp_net.check_if_paralell_lines(
                G_trolleybus, G_streets, crit_buffer=buffer_dist,
                min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='trolleybus')

        nodes, edges = hp_rw.nx_to_gdf(G_streets)

        # TODO: street edges with attributes to postgis
        edges.to_postgis("street_network_edges_with_attributes", postgis_connection, if_exists="replace")
        #edges.to_file(os.path.join(path_out_city, 'street_network_edges_with_attributes.shp'))

    print("assign_attributes_to_graph finished")

    # ==============================================================================
    # ---- Assign population density to network
    # ==============================================================================
    print("calculate_pop_density start")
    #if not os.path.exists(os.path.join(path_out_city, 'street_network_nodes_with_attributes_pop_density.shp')) or write_anyway:

    # Load facebook popluation data
    # TODO: get street edges with attributes from postgis
    gdf_street = edges
    #gdf_street = gpd.read_file(os.path.join(path_out_city, "street_network_edges_with_attributes.shp"))
    if G_streets:
        G = G_streets
    else:
        G = hp_rw.gdf_to_nx(gdf_street)

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
    #pop_pnts.to_file(os.path.join(path_out_city, "fb_pop.shp"))
    pop_pnts.to_postgis("fb_pop", postgis_connection, if_exists="replace")

    # ----Calculate population density
    G = hp_net.calc_edge_and_node_pop_density(
        G,
        pop_pnts,
        radius=radius_pop_density,
        attribute_pop='population',
        label='pop_den')

    # ----Calculate GFA density based on osm buildings
    #buildings_osm = gpd.read_file(os.path.join(path_out_city, "osm_buildings.shp"))
    G = hp_net.calc_GFA_density(G, osm_buildings_gdf, radius=radius_GFA_density, label='GFA_den')

    print("writing out shp")
    nodes, edges = hp_rw.nx_to_gdf(G)
    #nodes.to_file(os.path.join(path_out_city, 'street_network_nodes_with_attributes_pop_density.shp'))
    #edges.to_file(os.path.join(path_out_city, 'street_network_edges_with_attributes_pop_density.shp'))

    nodes.to_postgis("street_network_nodes_with_attributes_pop_density", postgis_connection, if_exists="replace")
    edges.to_postgis("street_network_edges_with_attributes_pop_density", postgis_connection, if_exists="replace")

print("-----finished script-----")
