"""Function to extract facebook data

https://data.humdata.org/dataset/united-states-high-resolution-population-density-maps-demographic-estimates

"""
import os
import geopandas as gpd
import rasterio
import rasterio.mask
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point
from progress.bar import Bar

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def get_fb_pop_data(gdf_bb, pop_crs, path_raw, label='population', buffer_distance=500):
    """Read rasterio file with population and mask
    bounding box (with buffer area) and get raster as points

    Note: Facebook data only comunit specificy, 100 m pixel (rough estimate)
    """
    src_shp = gdf_bb.crs

    # Buffer bounding box
    gdf_bb['geometry'] = gdf_bb.geometry.buffer(buffer_distance)

    gdf_bb_crs_bb = gdf_bb.to_crs(pop_crs)
    gdf_bb_crs_bb_geom = gdf_bb_crs_bb['geometry'].geometry[0]

    print("CRS BB:  {}".format(gdf_bb_crs_bb.crs))
    print("CRS pop: {}".format(pop_crs))
    assert int(gdf_bb_crs_bb.crs.srs.split(":")[1]) == pop_crs

    # Read data into shapefiles
    csv = pd.read_csv(path_raw, delimiter=",")

    print("Samlpe pop")
    print(csv.sample(n=10))
    if 'Lon' in csv.columns:
        lon = csv['Lon'].values
        lat = csv['Lat'].values
        pop = csv['Population'].values
    if 'longitude' in csv.columns:
        all_columns = list(csv.columns)
        all_columns.remove('longitude')
        all_columns.remove('latitude')
        name_pop = all_columns[0]
        print("Name pop: {}".format(name_pop))
        lon = csv['longitude'].values
        lat = csv['latitude'].values
        pop = csv[name_pop].values


    bar = Bar("Reading population", max=len(lon))
    population_list = []
    geometries = []

    cnt = 0
    for pop_val, lon_point, lat_point in zip(pop, lon, lat):
        pnt_to_check = Point(lon_point, lat_point)
        if pnt_to_check.intersects(gdf_bb_crs_bb_geom):  #within is slower
            population_list.append(pop_val)
            geometries.append(pnt_to_check)
        cnt += 1
        bar.next()
    bar.finish()

    assert len(population_list) > 0, "Wrong population datset"

    gpd_pop = gpd.GeoDataFrame(population_list, columns=[label], geometry=geometries, crs=pop_crs)
    gpd_pop = gpd_pop.reset_index(drop=True)

    print("Transform")
    gpd_pop = gpd_pop.to_crs(src_shp)

    return gpd_pop

def get_fb_pop_data_tif(gdf_bb, pop_crs, path_raw, path_temp, label='population', buffer_distance=500):
    """Read population from rasterio
    """

    # Buffer bounding box
    gdf_bb['geometry'] = gdf_bb.geometry.buffer(buffer_distance)

    #pop_crs = 4326
    #src_shp = 32616
    src_shp = int(gdf_bb.crs.srs.split(":")[1])
    gpf_bb = gdf_bb.to_crs('epsg:{}'.format(pop_crs))

    # Get clip geometries
    clip_geometry = getFeatures(gpf_bb)

    # Mask rasterio
    with rasterio.open(path_raw) as src:
        print("CRS of clip geometry: {}".format(gpf_bb.crs))
        print("CRS tif: {}".format(src.meta['crs'].data['init']))
        out_image, out_transform = rasterio.mask.mask(src, clip_geometry, crop=True)
        metadata = src.meta

        metadata.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})

    path_masked = os.path.join(path_temp, "masked_{}_{}.tif".format(out_image.shape[1], out_image.shape[2]))
    with rasterio.open(path_masked, "w", **metadata) as dest:
        dest.write(out_image)

    dataset_raw = rasterio.open(path_masked, driver='GTiff') 
    data = dataset_raw.read(1)

    # Convert rasterio to shapefile
    no_data_value = metadata['nodata']
    defined_nan_value = 0
    band1 = np.where(data == no_data_value, defined_nan_value, data)  # Replace placeholder

    # Get cell width and height
    pixel_width = list(out_transform)[0]
    pixel_height = list(out_transform)[4] * -1

    # Get coordinates top left
    top_left_x = dataset_raw.bounds.left
    top_left_y = dataset_raw.bounds.top

    height = band1.shape[0]
    widht = band1.shape[1]
    print("Width: {}  Height: {}".format(widht, height))

    coords = []
    value_list = []
    for j in range(height):
        for i in range(widht):
            value = band1[j, i]
            if value == defined_nan_value or np.isnan(value): # If empty, ignore
                pass
            else:
                x = top_left_x + (i * pixel_width) + pixel_width / 2
                y = top_left_y - (j * pixel_height) - pixel_height / 2
                coords.append(Point(x, y))
                value_list.append(value)

        shp_pop = gpd.GeoDataFrame(coords, columns=['geometry'], crs='epsg:{}'.format(pop_crs))
        shp_pop[label] = value_list

    # Transform to original crs
    shp_pop = shp_pop.to_crs('epsg:{}'.format(src_shp))

    return shp_pop


def filter_swiss_buildings(pop_pnts, bb_gdf):
    """Select only those ponts in extent
    """
    index_within = []
    geom_extent = bb_gdf.geometry[0]
    bar = Bar("Reading population", max=len(pop_pnts))
    for index_pop_pnt in pop_pnts.index:
        geom_pop = Point(pop_pnts.loc[index_pop_pnt].geometry)
        if geom_pop.within(geom_extent):
            index_within.append(index_pop_pnt)

        bar.next()
    bar.finish()
    pop_pnts = pop_pnts.loc[index_within]
    pop_pnts = pop_pnts.reset_index(drop=True)

    return pop_pnts
