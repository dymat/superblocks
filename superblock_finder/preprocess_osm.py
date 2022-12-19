"""
Download data from openstreetmap for superblock analysis

Note: Max 10,000 queries per day and download < 5 GB
https://wiki.openstreetmap.org/wiki/Overpass_API#:~:text=https%3A%2F%2Flz4.overpass%2Dapi.de%2Fapi%2Finterpreter&text=Any%20of%20the%20three%20servers,about%201%2C000%2C000%20requests%20per%20day.z
"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_superblocks)
import time
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

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
to_crs_meter_switzerland = 32632 # Target projection

path_temp = "/data/tmp"
path_out = "/data/cities"
path_pop_data = "/deu_pd_2020_1km.tif"  # Download data as outlined from data source in publication
write_anyway = False

# working steps
download_data = True
assign_attributes_to_graph = True
calculate_pop_density = True

hp_rw.create_folder(path_out)

# Window selection
length_in_m = 2000  # [m]
radius_pop_density = 100  # [m]
radius_GFA_density = 100 # [m]
sleep_time = 50
crit_bus_is_big_street = False
swiss_community = False

case_studies = [
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

for city in case_studies:
    path_out_city = os.path.join(path_out, str(city))
    city_metadata = hp_rw.city_metadata(city, path_pop_data=path_pop_data)
    to_crs_meter = city_metadata['crs']
    path_raw_pop = city_metadata['path_raw_pop']
    centroid_tuple = city_metadata['centroid_tuple']
    print("=== city: {}".format(city))

    if download_data:
        hp_rw.create_folder(path_out_city)

        # Download first bounding box shp (as WSG84 crs)
        centroid = Point(centroid_tuple)
        centroid = gpd.GeoDataFrame([centroid], columns=['geometry'], crs=crs_bb)
        centroid.to_file(os.path.join(path_out_city, "centroid.shp"))

        centroid_to_crs_meter = centroid.to_crs("epsg:{}".format(to_crs_meter))
        centroid.to_file(os.path.join(path_out_city, "centroid_{}.shp".format(to_crs_meter)))
        bb = hp_osm.BB(
            ymax=centroid_to_crs_meter.geometry.y[0] + length_in_m / 2,
            ymin=centroid_to_crs_meter.geometry.y[0] - length_in_m / 2,
            xmax=centroid_to_crs_meter.geometry.x[0] + length_in_m / 2,
            xmin=centroid_to_crs_meter.geometry.x[0] - length_in_m / 2)

        bb_gdf = bb.as_gdf(crs_orig=to_crs_meter)
        bb_gdf.to_file(os.path.join(path_out_city, "extent.shp"))
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
        if not os.path.exists(path_complete_streets_edges) or write_anyway:

            # Download buildings
            if not os.path.exists(path_buildings) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='buildings')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)

                    # Remove invalid polygons
                    osm_gdf = hp_osm.remove_faulty_polygons(osm_gdf)
                    osm_gdf.to_file(path_buildings)

            # Download water
            if not os.path.exists(path_water) or write_anyway:
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='water')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)
                        osm_gdf.to_file(path_water)

            # Download bus
            if not os.path.exists(path_bus) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bus')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)
                        osm_gdf.to_file(path_bus)

            # Download bridges
            if not os.path.exists(path_bridges) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='bridges')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf.to_file(path_bridges)

            # Download land use
            if not os.path.exists(path_landuse) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='landuse')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf.to_file(path_landuse)
            
            # Download tram
            if not os.path.exists(path_tram) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='tram')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf.to_file(path_tram)

            # Download bus
            if not os.path.exists(path_trolleybus) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='trolleybus')
                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    if osm_gdf.shape[0] > 0:
                        osm_gdf.to_file(path_trolleybus)
  
            # Download streets
            if not os.path.exists(path_streets_edges) or write_anyway:
                time.sleep(sleep_time)
                osm_gdf = hp_osm.overpass_osm(bb=bb, to_crs=to_crs_meter, extraction_type='streets')

                if osm_gdf.shape[0] > 0:
                    osm_gdf = hp_net.gdf_multilinestring_to_linestring(osm_gdf)  # Multiline gdf to singline gdf
                    osm_gdf = hp_net.clip_outer_polygons(osm_gdf, bb_geom)
                    osm_gdf = hp_net.remove_all_intersections(osm_gdf) # remove all intersections (lines which overlap)
                    osm_gdf = hp_net.remove_rings(osm_gdf)
                    G = hp_rw.gdf_to_nx(osm_gdf)
                    G_simple = hp_net.simplify_network(
                        G, crit_big_roads=False, crit_bus_is_big_street=crit_bus_is_big_street)

                    nodes, edges = hp_rw.nx_to_gdf(G_simple)
                    nodes.to_file(path_streets_nodes)
                    edges.to_file(path_streets_edges)

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

    if assign_attributes_to_graph:

        if not os.path.exists(os.path.join(path_out_city, 'street_network_edges_with_attributes.shp')) or write_anyway:
            # ==============================================================================
            # Assign  attributes from one graph to another graph
            # ==============================================================================
            path_tram = os.path.join(path_out_city, "tram.shp")
            path_bus = os.path.join(path_out_city, "bus.shp")
            path_street = os.path.join(path_out_city, "street_network_edges.shp")
            path_trolleybus = os.path.join(path_out_city, "trolleybus.shp")

            if os.path.exists(path_tram):
                gdf_tram = gpd.read_file(path_tram)
                G_tram = hp_rw.gdf_to_nx(gdf_tram)

            if os.path.exists(path_bus):
                gdf_bus = gpd.read_file(path_bus)
                G_bus = hp_rw.gdf_to_nx(gdf_bus)

            if os.path.exists(path_trolleybus):
                gdf_trolleybus = gpd.read_file(path_trolleybus)
                G_trolleybus = hp_rw.gdf_to_nx(gdf_trolleybus)

            if os.path.exists(path_street):
                gdf_streets = gpd.read_file(path_street)
                G_streets = hp_rw.gdf_to_nx(gdf_streets)
                

                # Remove footway (new)
                G_streets = hp_net.remove_edge_by_attribute(G_streets, attribute='tags.highw', value="footway")
                G_streets = hp_net.G_multilinestring_to_linestring(G_streets, single_segments=True)

                # Criteria which influcence how graphs are spatially merged
                buffer_dist = 10        # [m]
                min_edge_distance = 10  # [m]
                p_min_intersection = 80 # [m] minimum edge percentage which needs to be intersected

                G_streets = hp_net.simplify_network(
                    G_streets, crit_big_roads=False, crit_bus_is_big_street=crit_bus_is_big_street)

                nx.set_edge_attributes(G_streets, 0, 'tram')
                nx.set_edge_attributes(G_streets, 0, 'bus')
                nx.set_edge_attributes(G_streets, 0, 'trolleybus')

                if os.path.exists(path_bus):    
                    G_streets = hp_net.check_if_paralell_lines(
                        G_bus, G_streets, crit_buffer=buffer_dist,
                        min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='bus')

                if os.path.exists(path_tram):
                    G_streets = hp_net.check_if_paralell_lines(
                        G_tram, G_streets, crit_buffer=buffer_dist,
                        min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='tram')

                if os.path.exists(path_trolleybus):   
                    G_streets = hp_net.check_if_paralell_lines(
                        G_trolleybus, G_streets, crit_buffer=buffer_dist,
                        min_edge_distance=min_edge_distance, p_min_intersection=p_min_intersection, label='trolleybus')

                nodes, edges = hp_rw.nx_to_gdf(G_streets)
                edges.to_file(os.path.join(path_out_city, 'street_network_edges_with_attributes.shp'))

        print("assign_attributes_to_graph finished")

    # ==============================================================================
    # ---- Assign population density to network
    # ==============================================================================
    if calculate_pop_density:
        print("calculate_pop_density start")
        if not os.path.exists(os.path.join(path_out_city, 'street_network_nodes_with_attributes_pop_density.shp')) or write_anyway:

            # Load facebook popluation data
            gdf_street = gpd.read_file(os.path.join(path_out_city, "street_network_edges_with_attributes.shp"))
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
            pop_pnts.to_file(os.path.join(path_out_city, "fb_pop.shp"))

            # ----Calculate population density
            G = hp_net.calc_edge_and_node_pop_density(
                G,
                pop_pnts,
                radius=radius_pop_density,
                attribute_pop='population',
                label='pop_den')

            # ----Calculate GFA density based on osm buildings
            buildings_osm = gpd.read_file(os.path.join(path_out_city, "osm_buildings.shp"))
            G = hp_net.calc_GFA_density(G, buildings_osm, radius=radius_GFA_density, label='GFA_den')

            print("writing out shp")
            nodes, edges = hp_rw.nx_to_gdf(G) 
            nodes.to_file(os.path.join(path_out_city, 'street_network_nodes_with_attributes_pop_density.shp'))
            edges.to_file(os.path.join(path_out_city, 'street_network_edges_with_attributes_pop_density.shp'))

print("-----finished script-----")
