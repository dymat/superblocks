"""

#"""
import sys
import os
import pprint
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from networkx.generators.classic import ladder_graph
import numpy as np
import geopandas as gpd
from networkx.algorithms.operators.binary import intersection
from numpy.lib.function_base import extract
from rtree import index
import geopandas as gpd
from pandas.io.json import json_normalize
import requests
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from shapely import wkt
from progress.bar import Bar
from shapely.ops import snap

from superblocks.scripts.network import helper_read_write as hp_rw

'''def add_intersection(gdf):
    """Check if line segments, if yes, intersect split lines
    """
    assert gdf.index.is_unique

    rTree = index.Index()
    for index_nr, geometry in enumerate(gdf.geometry):
        rTree.insert(index_nr, geometry.bounds)

    index_with_new_geoms = {}
    for gdf_index in gdf.index:
        line1 = gdf.loc[gdf_index].geometry
        geometries_closeby = list(rTree.intersection(line1.bounds))

        for gdf_closeby in geometries_closeby:
            line2 = gdf.iloc[gdf_closeby].geometry.centroid

            if line1.crosses(line2):
                cross_pnts = line1.intersection(line2)

                if len(cross_pnts) == 1:
                    new_edge_start = line2.geometry.coords[0]
                    new_edge_end = line2.geometry.coords[-1]
                    new_edge_cross = cross_pnts[0]
                    new_edge = LineString(new_edge_start, new_edge_cross)
                    new_edge = LineString(new_edge_cross, new_edge_end)
                else:
                    raise Exception("more than one crossover")
  
    return gdf_with_intersection'''

def remove_buildng_plots(gdf_cadastre, gdf_osm_buildings):
    """Remove all polygons which have a building centroid on it
    """
    gdf_cadastre = gdf_cadastre.reset_index()

    rTree = index.Index()
    for index_nr, geometry in enumerate(gdf_osm_buildings.geometry):
        rTree.insert(index_nr, geometry.bounds)

    bar = Bar('Check if building centroid on plot:', max=gdf_cadastre.shape[0])

    index_to_remove = []
    for gdf_index in gdf_cadastre.index:
        geom_cad = gdf_cadastre.loc[gdf_index].geometry

        nr_of_centroids = 0
        geometries_closeby = list(rTree.intersection(geom_cad.bounds))

        for gdf_closeby in geometries_closeby:
            centroid = gdf_osm_buildings.iloc[gdf_closeby].geometry.centroid

            if centroid.within(geom_cad):
                nr_of_centroids += 1

        if nr_of_centroids > 0:
            index_to_remove.append(gdf_index)
        bar.next()
    bar.finish()
    print("Number of cadastre plots with buildings on it: {}".format(len(index_to_remove)))
    gdf_cadastre = gdf_cadastre.drop(index=index_to_remove)

    return gdf_cadastre


def mulitpolygon_to_singlepolygon(list_with_poly):

    singlepolygons = []
    for poly in list_with_poly:
        if poly.type == 'MultiPolygon':
            for i in poly:
                singlepolygons.append()
        elif poly.type == 'Polygon':
            singlepolygons.append(poly)
        else:
            raise Exception("Not polygon")

    return singlepolygons


def snap_geometries(geometries):
    """Iterate all nodes of geometries and snap
    
    square = Polygon([(1,1), (2, 1), (2, 2), (1, 2), (1, 1)])
    line = LineString([(0,0), (0.8, 0.8), (1.8, 0.95), (2.6, 0.5)])
    result = snap(line, square, 0.5)
    """
    snap_distance = 0.5 # [m]
    rTree = index.Index()
    for index_nr, geometry in enumerate(geometries):
        rTree.insert(index_nr, geometry.bounds)

    snapped_geometries = []

    for list_index, geometry in enumerate(geometries):
        geometry_to_snap = geometry
        geometries_closeby = list(rTree.intersection(geometry.bounds))

        for geometry_index in geometries_closeby:
            geometry_test_closeby = geometries[geometry_index]
            result = snap(geometry_to_snap, geometry_test_closeby, snap_distance)

            if result.equals(geometry_to_snap):
                pass
            else:
                geometry_to_snap = result

        snapped_geometries.append(geometry_to_snap)

    return snapped_geometries


def iterative_merge(polygons_to_add):
    """Clip overlapping into individual polygons

    Converts the following lines into an itterative process

        merged = linemerge([i.boundary for i in polygons_to_add])
        borders = unary_union(merged)
        polygons = polygonize(borders)

    Note: By using exterior instead of boundary, it removes holes
    in polygons
    """
    poly_to_add = []
    for cnt, polygon in enumerate(polygons_to_add):
        if cnt > 0:
            poly_to_add.append(polygon)
            borders_to_add = [i.exterior for i in poly_to_add]
            merged = linemerge(borders_to_add)
            borders = unary_union(merged)

            poly_to_add = list(polygonize(borders))
        elif cnt == 0:
            poly_to_add.append(polygon)
        else:
            pass

    # Polygonize
    #polygons = polygonize(poly_to_add)
    #polygons = polygonize(borders)

    return poly_to_add


def cut_polygon_by_line(polygon, line):
    """Cut polygon by line
    """
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)

    return list(polygons)


def clip_cadastre_too_far_away(
        intervention_id_gdf,
        gdf_cadastre_orig,
        large_buffer_dist,
        rTree_cadastre
    ):
    """

    """
    gdf_cadastre = gdf_cadastre_orig.copy()
    gdf_intervention = intervention_id_gdf.copy()

    # Buffer superblock streets
    gdf_intervention['geometry'] = gdf_intervention.geometry.buffer(large_buffer_dist)
    intervention_buffer_beom = unary_union(gdf_intervention.geometry.tolist())

    potential_intersections = rTree_cadastre.intersection(intervention_buffer_beom.bounds)

    # Interate cadastre and clip if intersection
    index_to_keep = []
    for potential_intersection in potential_intersections:
        cadastre_geom = gdf_cadastre.loc[potential_intersection].geometry
        if cadastre_geom.intersects(intervention_buffer_beom):
            clipped_within_buffer = cadastre_geom.intersection(intervention_buffer_beom)
            gdf_cadastre.at[potential_intersection, 'geometry'] = clipped_within_buffer
            index_to_keep.append(potential_intersection)

    # Remove all non-intersecting geometris
    gdf_cadastre_clipped = gdf_cadastre.loc[index_to_keep]
    gdf_cadastre_clipped = gdf_cadastre_clipped.reset_index(drop=True)

    return gdf_cadastre_clipped


def spatially_refine_cadastre_with_osm(
        gdf_edges_superblock,
        G_edges_superblock,
        gdf_cadastre,
        offset_distance=15,
        crit_intersection_distance=5
    ):
    """Iterate superblock edges and for each node on line,
    create "cutter - lines" which are perpendicular to the lines
    which  can then be used to split cadastre plots

    https://stackoverflow.com/questions/57065080/draw-perpendicular-line-of-fixed-length-at-a-point-of-another-line

    """
    assert gdf_cadastre.index.is_unique

    cutter_lines = []

    # Remove all lines with zero length and reset index
    gdf_edges_superblock = gdf_edges_superblock[gdf_edges_superblock.geometry.length > 0]
    gdf_edges_superblock = gdf_edges_superblock.reset_index(drop=True)

    # Replace all polygons with exterior polygon (i.e. remove holes)
    for index_cadastre in gdf_cadastre.index:
        polygon = gdf_cadastre.loc[index_cadastre].geometry.exterior
        gdf_cadastre.at[index_cadastre, 'geometry'] = Polygon(polygon)

    gdf_edges_superblock = gdf_edges_superblock.reset_index(drop=True)

    # Offset distance (right and left perpendicular distance)
    #for index_edge in gdf_edges_superblock.index:
    #    edge_geom = gdf_edges_superblock.loc[index_edge].geometry
    for edge in G_edges_superblock.edges:
        edge_geom = G_edges_superblock.edges[edge]['geometry']
        x_list = list(edge_geom.coords.xy[0])
        y_list = list(edge_geom.coords.xy[1])

        if edge_geom.is_closed:  # If closed, remove last point
            x_list = x_list[:-1]
            y_list = x_list[:-1]

        if edge_geom.geom_type == 'LineString':
            start_node = Point(x_list[0], y_list[0])
            ende_node = Point(x_list[-1], y_list[-1])

            geom_direction = LineString([start_node, ende_node])
            geom_inverse = LineString([ende_node, start_node])

            # NEw: Only create clip line if degree == 1 (not both ends but only outer end)
            degree_geom_direction = G_edges_superblock.degree[(x_list[0], y_list[0])]
            degree_geom_inverse = G_edges_superblock.degree[(x_list[-1], y_list[-1])]
            if degree_geom_direction <= 2 and degree_geom_inverse <= 2:
                lines_to_create_clipline = [geom_direction, geom_inverse]
            elif degree_geom_direction <= 2:
                lines_to_create_clipline = [geom_inverse]
            elif degree_geom_inverse <= 2:
                lines_to_create_clipline = [geom_direction]
            else:
                lines_to_create_clipline = []

            for geom in lines_to_create_clipline:
                left = geom.parallel_offset(offset_distance, 'left')
                right = geom.parallel_offset(offset_distance, 'right')
                c = left.boundary[1]
                d = right.boundary[0]  # note the different orientation for right offset
                cd = LineString([c, d])
                cutter_lines.append(cd)
        elif edge_geom.geom_type == 'MultiLineString':
            # Based on first two last points
            start_node = Point(x_list[0], y_list[0])
            ende_node = Point(x_list[1], y_list[1])

            # New: Only create clip line if degree <= 2
            degree_geom_direction = G_edges_superblock.degree[(x_list[0], y_list[0])]
            degree_geom_inverse = G_edges_superblock.degree[(x_list[1], y_list[1])]
            start = False
            end = False
            if degree_geom_direction <= 2 and degree_geom_inverse <= 2:
                start = True
                end = True
            elif degree_geom_direction <= 2:
                start = False
                end = True
            elif degree_geom_inverse <= 2:
                start = True
                end = False
            else:
                pass

            if start:
                geom = LineString([start_node, ende_node])
                left = geom.parallel_offset(offset_distance, 'left')
                right = geom.parallel_offset(offset_distance, 'right')
                c = left.boundary[1]
                d = right.boundary[0]  # note the different orientation for right offset
                cd = LineString([c, d])
                cutter_lines.append(cd)
            if end:
                # Based on second two last points
                start_node = Point(x_list[-2], y_list[-2])
                ende_node = Point(x_list[-1], y_list[-1])
                geom = LineString([start_node, ende_node])
                left = geom.parallel_offset(offset_distance, 'left')
                right = geom.parallel_offset(offset_distance, 'right')
                c = left.boundary[1]
                d = right.boundary[0]  # note the different orientation for right offset
                cd = LineString([c, d])
                cutter_lines.append(cd)
        else:
            raise Exception("Wrong geom type")

    cutterlines = gpd.GeoDataFrame(cutter_lines, columns=['geometry'], crs=gdf_edges_superblock.crs)
    #cutterlines.to_file("C:/_scrap/spark/cutterlines.shp")

    # --------------------------------
    # 1. Remove all cutterlines which don't intersect 
    # 2. Remove all intersecting cutterlines (anyone)
    # --------------------------------
    print("... creating search tree for all cutting lines")
    gdf_cadastre = gdf_cadastre.reset_index(drop=True)
    rTree = index.Index()
    for iloc, cadastre_plot in enumerate(gdf_cadastre.geometry):
        rTree.insert(iloc, cadastre_plot.bounds)

    '''TODO:# (1) get only cutteryline which cut actually a polygon
    cutter_lines_only_insterction = []
    for cutter_line in cutter_lines:
        plots_intersect = list(rTree.intersection(cutter_line.bounds))  # Split polygon
        for plot_intersect_index in plots_intersect:
            cutter_lines_only_insterction.append(cutter_line)

    # (2) remove intersecting cutting lines
    rTree_lines = index.Index()
    for iloc, cut_line in enumerate(cutter_lines_only_insterction.geometry):
        rTree.insert(iloc, cut_line.bounds)

    cutter_lines_unique = []
    cutter_line_centroid = []

    tested = []
    for cutter_line in cutter_lines_only_insterction:
        if cutter_line.bounds not in tested:
            crit_unique = True
            close_lines = rTree_lines.intersection(cutter_line.bounds)
            for close_line in close_lines:
                if close_line.intersects(cutter_line):
                    crit_unique = False
                    break
            if crit_unique:
                cutter_lines_unique.append(cutter_line)
            else:
                tested.append(cutter_line.bounds)

    cutter_lines = cutter_lines_unique
    '''

    # ---------------------------------------
    # Use cutter lines to intersect polygons
    # ---------------------------------------
    poly_index_to_del = []
    polygons_to_add = []

    # Iterate cut lines
    for cutter_line in cutter_lines:
        plots_intersect = list(rTree.intersection(cutter_line.bounds))  # Split polygon
        for plot_intersect_index in plots_intersect:
            plot_intersect = gdf_cadastre.iloc[plot_intersect_index].geometry
            if plot_intersect.intersects(cutter_line):

                # Check if intersecty by considerable length
                intersection_distance = 0
                intersection_edge = plot_intersect.intersection(cutter_line)
                if intersection_edge.type == 'LineString':
                    intersection_distance += intersection_edge.length
                elif intersection_edge.type == 'MultiLineString':
                    for i in intersection_edge:
                        intersection_distance += intersection_edge.length
                else:
                    raise Exception("Something went wrong here")

                if intersection_distance > crit_intersection_distance:
                    cut_poly = list(cut_polygon_by_line(plot_intersect, cutter_line))

                    if len(cut_poly) > 1:
                        # IF not very small polygons are created, add cutted line
                        poly_index_to_del.append(plot_intersect_index)
                        for i in cut_poly:
                            polygons_to_add.append(i)
                    else:
                        pass # no cut

    # NEW HERE
    gdf_cadastre = gdf_cadastre.drop(index=poly_index_to_del)

    # Because many individual polygons which are intersected several times clean up and remove overlaps
    previous = gpd.GeoDataFrame(polygons_to_add, columns=['geometry'])
    merged_previous = previous.unary_union

    # ---- TRY SNAPPING POLYGONS
    # Snap all lines which are nearly identical to improve polygonization
    polygons_to_add_no_holes = snap_geometries(polygons_to_add)
    #_a = gpd.GeoDataFrame(polygons_to_add_no_holes, columns=['geometry'])
    #_a.to_file("C:/_scrap/spark/polygons_to_add_no_holes.shp")

    merged = linemerge([i.boundary for i in polygons_to_add_no_holes])
    borders = unary_union(merged)
    new_cut_polygons = list(polygonize(borders))
    new_cut_polygons_gdf = gpd.GeoDataFrame(new_cut_polygons, columns=['geometry'])

    # Clip away everything in new which is outside of the old and very small sliver polygons
    crit_minimum_area_overlap = 0.5   # [m2]

    for new_cut_polygon in new_cut_polygons_gdf.index:
        new_geom = new_cut_polygons_gdf.loc[new_cut_polygon].geometry

        # Try clipping away
        try:
            within_old = new_geom.intersection(merged_previous)
        except:
            print("INfo: somethign went wront with intersection")
            continue

        # If Multipolygon remove very small sliver polygons
        if within_old is None:
            within_old = None
        elif within_old.type == 'LineString':
            within_old = None
        elif within_old.type == 'Point':
            within_old = None
        elif within_old.type == 'Polygon':
            if within_old.area > crit_minimum_area_overlap:
                within_old = within_old
            else:
                within_old = None
        elif within_old.type != 'Polygon':
            within_old_only_polygon = []
            for entry in within_old:
                if entry.type == 'Polygon':
                    if entry.area > crit_minimum_area_overlap:
                        within_old_only_polygon.append(entry)
            within_old_only_polygon = unary_union(within_old_only_polygon)

            if within_old_only_polygon.area < crit_minimum_area_overlap:
                # Inner faulty polygon to remove
                within_old = None
            else:
                within_old = within_old_only_polygon
        else:
            raise Exception("type unclear")

        new_cut_polygons_gdf.at[new_cut_polygon, 'geometry'] = within_old

    # Remove None polygons
    new_cut_polygons_gdf = new_cut_polygons_gdf.loc[new_cut_polygons_gdf.geometry != None]
    new_cut_polygons_gdf = new_cut_polygons_gdf.reset_index(drop=True)
    #new_cut_polygons_gdf.to_file("C:/_scrap/spark/nachher2.shp")

    # Append polygons
    gdf_cadastre = gdf_cadastre.append(new_cut_polygons_gdf)

    # ---Drop very large polygons
    max_polygon_size = 10000  # [m2]
    gdf_cadastre = gdf_cadastre.loc[gdf_cadastre.geometry.area < max_polygon_size]
    gdf_cadastre = gdf_cadastre.reset_index(drop=True)

    # Fix columns
    gdf_cadastre['cad_id'] = range(gdf_cadastre.shape[0])
    gdf_cadastre = gdf_cadastre[['geometry', 'cad_id']]

    # ---Replace multipolygon and delete points and lines
    index_to_drop = []
    new_polygons = gpd.GeoDataFrame(columns=['geometry'])
    for i in gdf_cadastre.index:
        geometry = gdf_cadastre.loc[i].geometry
        geom_type = geometry.type
        if geom_type == 'Point' or geom_type == 'LineString' or geometry.bounds == ():
            index_to_drop.append(i)
        else:
            if geom_type == 'GeometryCollection':
                index_to_drop.append(i)
                for single_element in geometry:
                    if single_element.type == 'Polygon':
                        new_polygons.append(single_element)

    gdf_cadastre = gdf_cadastre.drop(index=index_to_drop)
    gdf_cadastre = gdf_cadastre.append(new_polygons)

    # ---Remove faulty polygons from cadastre
    index_to_drop = []
    for i in gdf_cadastre.index:
        if gdf_cadastre.loc[i].geometry is None:
            index_to_drop.append(i)
    gdf_cadastre = gdf_cadastre.drop(index=index_to_drop)
    gdf_cadastre = gdf_cadastre.reset_index(drop=True)

    return gdf_cadastre


def split_edges_with_polygons(gdf_intervention, cadastre):
    """
    If a line intersects a polygon, create multipe
    lines
    """
    rTree = index.Index()
    for edge_nr, cadastre_plot in enumerate(cadastre.geometry):
        rTree.insert(edge_nr, cadastre_plot.bounds)

    edges_to_drop = []
    edges_to_add = []
    for index_edge in gdf_intervention.index:
        edge_geometry = gdf_intervention.loc[index_edge].geometry

        # Test if intersect
        plots_intersect = list(rTree.intersection(edge_geometry.bounds))
        for plot_intersect_iloc in plots_intersect:
            cadastre_plot_geom = cadastre.iloc[plot_intersect_iloc].geometry
            if cadastre_plot_geom.intersects(edge_geometry):
                part_overlap = edge_geometry.intersection(cadastre_plot_geom)
                part_intersection = edge_geometry.difference(cadastre_plot_geom)

                added = False
                if part_overlap.type == 'LineString':
                    edges_to_add.append(part_overlap)
                    added = True
                if part_intersection.type == 'LineString':
                    edges_to_add.append(part_intersection)
                    added = True
                if added:
                    edges_to_drop.append(index_edge)

    gdf_intervention = gdf_intervention.drop(index=edges_to_drop)
    gdf_new = gpd.GeoDataFrame(edges_to_add, columns=['geometry'], crs=gdf_intervention.crs)
    gdf_intervention = gdf_intervention.append(gdf_new)

    return gdf_intervention


def spatial_select_cadastre(
        gdf_intervention,
        cadastre,
        rTree,
        crit_length_intersection,
        buffer_size=15,
        crit_area_intersection=0.5,
        min_area_of_cadastre_to_check_intersection=20
    ):
    """Select cadastre polygons based on roads
    """
    #min_area_of_cadastre_to_check_intersection = 20  # m
    cadastre_plot_index = []
    gdf_intervention_buffer = gdf_intervention.buffer(buffer_size)
    united = unary_union(gdf_intervention_buffer.geometry.tolist())

    if united.type == 'MultiPolygon':
        united = [i for i in united.geoms]
    else:
        united = [united]

    gdf_intervention_union = gpd.GeoDataFrame(united, columns=['geometry'], crs=gdf_intervention.crs)

    if gdf_intervention_union.shape[0] > 1:
        # Select largest
        all_areas = list(gdf_intervention_union.geometry.area)
        index_max_area = all_areas.index(max(all_areas))
        gdf_intervention_union = gdf_intervention_union.iloc[[index_max_area]]

    assert gdf_intervention_union.shape[0] == 1

    gdf_intervention_union_geom = gdf_intervention_union.geometry.tolist()[0]
    plots_intersect = list(rTree.intersection(gdf_intervention_union_geom.bounds))

    for plot_intersect in plots_intersect:
        cadastre_plot_geom = cadastre.iloc[plot_intersect].geometry
        if gdf_intervention_union_geom.contains(cadastre_plot_geom):
            cadastre_plot_index.append(plot_intersect)     
        elif cadastre_plot_geom.intersects(gdf_intervention_union_geom):

            # If too small polygon, do not consider (the very small ones within are contained)
            if cadastre_plot_geom.area > min_area_of_cadastre_to_check_intersection: 
                intersection_polygon = cadastre_plot_geom.intersection(gdf_intervention_union_geom)
                p_intersection_area = intersection_polygon.area / cadastre_plot_geom.area
                if p_intersection_area > crit_area_intersection:
                    cadastre_plot_index.append(plot_intersect)
        else:
            pass                                

    '''
    cadastre_plot_index = []
    #added_indexs = []
    for index_edge in gdf_intervention.index:
        edge_geometry = gdf_intervention.loc[index_edge].geometry
        edge_length = edge_geometry.length
        ##edge_geometry = edge_geometry.buffer(1, cap_style=2)
        edge_large_buffer = edge_geometry.buffer(buffer_size)

        # Test if intersect
        plots_intersect = list(rTree.intersection(edge_geometry.bounds))
        for plot_intersect_iloc in plots_intersect:
            cadastre_plot_geom = cadastre.iloc[plot_intersect_iloc].geometry

            #crit_add_cadastre = False
            #if cadastre_plot_geom.contains(edge_geometry): 
            #    crit_add_cadastre = True
            #elif cadastre_plot_geom.intersects(edge_geometry):
            #    crit_add_cadastre = True

            # Test if polygon fully in large buffered (to clip away 
            # street outside the superblock whith wohly small line lends itnersecting)
            ##if not edge_large_buffer.contains(cadastre_plot_geom):
            ##    crit_add_cadastre = False

            #if crit_add_cadastre:
            #    if plot_intersect_iloc not in added_indexs:
            #        cadastre_plot_index.append(plot_intersect_iloc) 
            #        added_indexs.append(plot_intersect_iloc)


            length_intersection = 0
            crit_add_cadastre = False
            # Check if polygon contains road
            if cadastre_plot_geom.contains(edge_geometry):
                crit_add_cadastre = True
            # Check if small fully within buffer
            elif edge_large_buffer.contains(cadastre_plot_geom):
                crit_add_cadastre = True  
            elif cadastre_plot_geom.intersects(edge_geometry):
                geom_intersection = cadastre_plot_geom.intersection(edge_geometry)

                if geom_intersection.geom_type == 'GeometryCollection':
                    for line_intersection in geom_intersection:
                        if line_intersection.geom_type == 'LineString':
                            length_intersection += line_intersection.length
                else:  
                    if geom_intersection.geom_type == 'LineString':
                        length_intersection = geom_intersection.length
                    elif geom_intersection.geom_type == 'MultiLineString':
                        for line_intersection in geom_intersection:
                            length_intersection += line_intersection.length
                    elif geom_intersection.geom_type == 'Point':
                        pass
                    else:
                        raise Exception("Error in intersection {}".format(geom_intersection.geom_type))
            else:
                pass # not within
  
            # Percentage of interseciton line in polygon
            intersection_p = length_intersection / edge_length
            #if crit_add_cadastre or length_intersection > crit_length_intersection:
            if crit_add_cadastre or intersection_p > crit_length_intersection:
                cadastre_plot_index.append(plot_intersect_iloc)                                  
    '''
    # Select all cadastre plots which are contained or intersected (> crit)
    cadastre = cadastre.iloc[cadastre_plot_index]

    # Merge all street polygons
    buffer_clean = 0.2
    merged_areas = unary_union(cadastre.geometry.tolist())
    merged_areas = merged_areas.buffer(buffer_clean)
    merged_areas = merged_areas.buffer(buffer_clean * -1)
    if merged_areas.type == 'MultiPolygon':
        # Only consider smalles area
        merged_areas = hp_rw.get_largest_polygon(merged_areas)

    out_gdf = gpd.GeoDataFrame([merged_areas], columns=['geometry'], crs=cadastre.crs)

    # Must be a single polygon
    assert out_gdf.shape[0] == 1, "size: {}".format(out_gdf.shape)

    return out_gdf


class BB(object):
    def __init__(self, ymax, ymin, xmax, xmin):
        """Constructor of bounding box
        """
        self.ymax = ymax
        self.ymin = ymin
        self.xmax = xmax
        self.xmin = xmin

    def as_coordinates(self):
        return ((self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax))

    def as_gdf(self, crs_orig):
        coords = ((self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax))
        gdf_bb = gpd.GeoDataFrame(geometry=[Polygon(coords)])
        gdf_bb.crs = "epsg:{}".format(crs_orig)
        return gdf_bb


def remove_faulty_polygons(gdf):
    """Remove all geometry whcih are strange
    """
    assert gdf.index.is_unique

    index_to_delete = []
    for i in gdf.index:
        geom = gdf.loc[i].geometry
        if (geom.is_valid) and (
            geom.bounds != ()) and (
                geom.wkt != 'EMPTY') and (geom.bounds != None):
                pass
        else:
            index_to_delete.appned(index_to_delete)
            
            if geom == None:
                raise Exception("fff {}".format(geom))

    print("Number of faulty geoms to delete: {}".format(len(index_to_delete)))
    gdf = gdf.drop(index=index_to_delete)

    gdf = gdf.reset_index(drop=True)
    return gdf


def overpass_osm(
        bb,
        to_crs,
        extraction_type,
    ):
    """Qury overpass and extract based on command

    http://overpass-turbo.eu Overpass query
    https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_API_by_Example

    Note: For OVerpass, feed in 4326 crs
    streets
    bus
    tram
    residential_area
    """
    print("Extraction code for : {}".format(extraction_type))
    osm_crs = 4326  # 3857 https://wiki.openstreetmap.org/wiki/Converting_to_WGS84

    bb_coordinates = "({},{},{},{})".format(bb.ymin, bb.xmin, bb.ymax, bb.xmax)

    if extraction_type == 'streets':
        """Get selected streets
        """
        #way["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|service|residential|unclassified|living_street|pedestrian|track"]({{bbox}});
        highway_types_to_merge = [
            'unclassified',
            'primary',
            'residential',
            'tertiary',
            'secondary',
            'secondary_link',
            'service',
            'construction',
            'living_street',
            'footway',
            'trunk',
            #'trunk_link' #
            
            ]
        str_to_enter = "'highway'~'"
        for i in highway_types_to_merge:
            str_to_enter = str_to_enter + "{}|".format(i)
        str_to_enter = str_to_enter[:-1]
        str_to_enter += "'"

        """Get all streets as lines"""
        query = """
            [out:json];
            (
            way[{}]{};
            );
            out geom;
            """.format(str_to_enter, bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'all_streets':
        """Get all streets as lines"""
        query = """
            [out:json];
            (
            way["highway"]{};
            );
            out geom;
            """.format(bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'water':
        query = """
            [out:json];
            (
            //way["natural"="water"]{};//
            relation["natural"="water"]{};
            );
            out geom;
            """.format(bb_coordinates, bb_coordinates)
        geom_type = 'Polygon'
    elif extraction_type == 'landuse':
        query = """
            [out:json];
            (
            way['landuse']{};
            relation['landuse']{};
            );
            out geom;
            """.format(bb_coordinates, bb_coordinates)
        geom_type = 'Polygon'
    elif extraction_type == 'residential_area':
        """Get all residential areas"""
        query = """
            [out:json];
            (
            way['landuse' = 'residential']{};
            );
            out geom;
            """.format(bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'tram':
        """Get all trams"""
        query = """
            [out:json]; //[timeout:100]
            (
            // node["railway"]{};
            way["railway"="tram"]{};
            // relation["railway"]{};
            );
            out geom;""".format(bb_coordinates, bb_coordinates, bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'bridges':
        """Get all bridgess"""
        query = """
            [out:json]; //[timeout:100]
            (
            // node["bus"]{};
            way["man_made"="bridge"]{};
            relation["man_made"="bridge"]{};
            );
            out geom;""".format(bb_coordinates, bb_coordinates, bb_coordinates)
        geom_type = 'Polygon'
    elif extraction_type == 'bus':
        """Get all bus lanes"""
        query = """
            [out:json]; //[timeout:100]
            (
            // node["bus"]{};
            way["route"="bus"]{};
            relation["route"="bus"]{};
            );
            out geom;""".format(bb_coordinates, bb_coordinates, bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'trolleybus':
        """Get all trolleybus lanes"""
        query = """
            [out:json]; //[timeout:100]
            (
            // node["bus"]{};
            //way["route"="trolleybus"]{};
            relation["route"="trolleybus"]{};
            );
            out geom;""".format(bb_coordinates, bb_coordinates, bb_coordinates)
        geom_type = 'LineString'
    elif extraction_type == 'buildings':
        """Get all buildings"""
        query = """
            [out:json];
            (
            way["building"]{};
            relation["building"]{};
            );
            out geom;""".format(bb_coordinates, bb_coordinates)
        geom_type = 'Polygon'
    else:
        raise Exception("Please define correct overpassturbo keyword")
    # Overpass API URL
    url = 'http://overpass/api/interpreter'
    #url = 'http://overpass-api.de/api/interpreter'

    r = requests.get(
        url,
        params={'data': query})

    if r.reason != 'OK':
        print("Status code: {}".format(r.reason))
        raise Exception("Overpass Turbo request failed: {}".format(r.reason))

    # Read response as JSON and get the data
    data = r.json()['elements']

    # Create a DataFrame from the data
    df = json_normalize(data)

    try:
        df = df.rename(columns={"tags.oneway": "tags.one"})
    except:
        pass

    # -----Intial empty gdf
    gdf = gpd.GeoDataFrame()
    if df.shape[0] == 0:
        print("No data available for download for {}".format(extraction_type))
    else:
        gdf_rel_and_way = gpd.GeoDataFrame()
        df_relation = df[df['type'] == 'relation']
        df_way = df[df['type'] == 'way']

        # -------Convert relation to geometries
        if df_relation.shape[0] > 0:

            # Add members
            geometry_list = []
            attribute_list = []
            for i in df_relation.index:
                attributes = df_relation.loc[i]
                for member in df_relation.loc[i].members:
                    if member['type'] == 'node':
                        pass
                    elif member['type'] == 'way':
                        if any(isinstance(i, list) for i in member['geometry']):  # IF nested list
                            for j in member['geometry']:
                                geometry_list.append(j)
                                attribute_list.append(attributes)
                        else:
                            geometry_list.append(member['geometry'])
                            attribute_list.append(attributes)

            df_relation_all_members = gpd.GeoDataFrame(attribute_list)
            df_relation_all_members = df_relation_all_members.drop(columns=['members'])
            df_relation_all_members['geometry'] = geometry_list

            gdf = df_relation_all_members[df_relation_all_members['geometry'].notna()]
            gdf_rel_and_way = gdf_rel_and_way.append(gdf)
            gdf_rel_and_way = gdf_rel_and_way.reset_index(drop=True)

        # -------Add way objects
        if df_way.shape[0] > 1:
            gdf_rel_and_way = gdf_rel_and_way.append(df_way)
            gdf_rel_and_way = gdf_rel_and_way.reset_index(drop=True)

        if gdf_rel_and_way.shape[0] > 0:
            # Iterate row (each row is a polygon) and create geometry
            geometry_list = []
            attribute_list = []
            for i in gdf_rel_and_way.index:
                attribute_entry = gdf_rel_and_way.loc[i]
                lat_point_list = [j['lat'] for j in gdf_rel_and_way.loc[i].geometry]
                lon_point_list = [j['lon'] for j in gdf_rel_and_way.loc[i].geometry]
                if geom_type == 'Polygon' and len(lat_point_list) > 2:  # Polygon has min 3 points
                    geom = Polygon(zip(lon_point_list, lat_point_list))

                    if not geom.is_valid:  # Try to fix polygon
                        geom = geom.buffer(0)
                    if geom.is_valid:
                        attribute_list.append(attribute_entry)
                        geometry_list.append(geom)
                    else:
                        pass  # not valid polygon
                elif geom_type == 'LineString' and len(lat_point_list) > 1:
                    polygon_geom = LineString(zip(lon_point_list, lat_point_list))
                    attribute_list.append(attribute_entry)
                    geometry_list.append(polygon_geom)
                elif geom_type == 'Point':
                    polygon_geom = Point(zip(lon_point_list, lat_point_list))
                    attribute_list.append(attribute_entry)
                    geometry_list.append(polygon_geom)
                else:
                    pass #raise Exception("Not correct geometry defined {}".format(geom_type))

            # ---Create gdf and assign projection
            gdf = gpd.GeoDataFrame(attribute_list)
            gdf = gdf.drop(columns=['geometry'])
            gdf['geometry'] = geometry_list

            columns_to_drop = ['nodes', 'bounds.maxlon', 'bounds.maxlat', 'bounds.minlon', 'bounds.minlat', 'id']
            for column in columns_to_drop:
                if column in gdf.columns.tolist():
                    gdf = gdf.drop(columns=[column])

            # Transform
            gdf = gpd.GeoDataFrame(gdf, geometry=geometry_list, crs="EPSG:{}".format(osm_crs)) # quick fix, since gdf.drop() returns a pandas dataframe instead of a geodataframe
            gdf.to_crs(f"epsg:{to_crs}", inplace=True)

            # Clean streets (Remove tunnels)
            try:
                # Remove tunnels which are longer than 300 m
                min_dist = 300
                tunnel_to_remove = gdf[(gdf['tags.tunnel'] == 'yes') & (gdf['geometry'].length > min_dist)].index.tolist()
                gdf = gdf.drop(index=tunnel_to_remove)
            except:
                pass

    return gdf
