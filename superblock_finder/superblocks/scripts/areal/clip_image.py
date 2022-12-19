"""Load Tif and clip

https://gis.stackexchange.com/questions/244667/cropping-area-of-interest-from-raster-file-using-python

https://www.youtube.com/watch?v=3kj8uoOlwjg&ab_channel=HatariLabs

reprojection
https://rasterio.readthedocs.io/en/latest/topics/reproject.html


https://rasterio.readthedocs.io/en/latest/quickstart.html#

"""
import os
import sys
import rasterio.mask
import pandas as pd
import geopandas as gpd
import json
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

# Imports
from superblocks.scripts.network import helper_osm as hp_osm

# CRS of SWISSIMAGE
src_tif = 2056 # CH1903+   # WSG 84: 4326
src_bb = 4326
#dst_crs = 'epsg:{}'.format(src_tif)

file_name = 'DOP10_LV95-2570_1260-2017-1_17.tif'#Note: replace - with _
path_tif = "K:/superblocks/01-data/_image/{}".format(file_name)
path_tif_attributed = "K:/superblocks/01-data/_image/_reproject_attributed.tif"
path_tif_clipped = "K:/superblocks/01-data/_image/_temp_clipped_tif.tif"
path_metadata = "K:/superblocks/01-data/_image/DOP10_LV95_Actual.csv"
path_test_clip_polygon = "K:/superblocks/01-data/_image/test_clipper.shp"

# Get top left coordinate from metadata
metadata = pd.read_csv(path_metadata, delimiter=';')

x_topleft = metadata[metadata['Filename'] == file_name[:-4].replace("-", "_") ].UpperLeftX.tolist()[0]
y_topleft = metadata[metadata['Filename'] == file_name[:-4].replace("-", "_") ].UpperLeftY.tolist()[0]

# Transform SWISSIMAGE CRS (2056) to WSG 84 (4326)
wgs2056_top_left = Point(x_topleft, y_topleft)
wgs2056_top_right = Point(x_topleft + 1000, y_topleft)
wgs2056 = pyproj.CRS('epsg:{}'.format(2056))
wgs4326 = pyproj.CRS('epsg:{}'.format(4326))
project = pyproj.Transformer.from_crs(wgs2056, wgs4326, always_xy=True).transform
utm_point = transform(project, wgs2056_top_left)
utm_point_top_right = transform(project, wgs2056_top_right)
m_1000_in_degree = utm_point_top_right.x - utm_point.x
print("Coordinates in crs_2056:         {}  {}".format(utm_point.x, utm_point.y))
print("Distance of 1000m in utm-8:          {}".format(m_1000_in_degree))

# Calculate transformation
# https://rasterio.readthedocs.io/en/latest/quickstart.html
transform = Affine(
    1, 0, x_topleft,
    0, -1, y_topleft)
nr_of_cells = 10000  # [SWISSIMAGE 10cm : 10'000 pixels]

# ------------------------------------------------------------------------
# Read with rasterio (somehow CRS does not work properly)
# ------------------------------------------------------------------------
dataset_no_attributed = rasterio.open(path_tif)
data_raw = dataset_no_attributed.read()

# Open TIF with missing projection and add correct projection
with rasterio.open(
    path_tif_attributed,
    'w',
    driver='GTiff',
    height=nr_of_cells,
    width=nr_of_cells,
    count=3,
    dtype='uint8',
    crs='epsg:{}'.format(src_tif),
    transform=transform,
) as dataset:
    array = data_raw
    print("----- Info -----")
    print(dataset.crs)
    print(dataset.transform)
    print(dataset.bounds)
    print(array.shape)
    print("---")
    # Writing out tif again with correct projection
    dataset.write(array)
    #dataset.close()
    #dataset.open()

# Load with full properties
'''dataset2 = rasterio.open(
    path_tif,
    'w',
    driver='GTiff',
    height=nr_of_cells,
    width=nr_of_cells,
    count=3,
    dtype='uint8',
    crs='epsg:{}'.format(src_tif),
    transform=transform)
print("--- TIF Info ---")
print(dataset2.crs)
print(dataset2.transform)
print(dataset2.bounds)
print("---")
dataset.write(dataset_no_attributed_data)'''

# ----------------
# Define clipping bounding box
# ------------
'''ymin = utm_point.y - (m_1000_in_degree / 2)     # 47.3601065
ymax = utm_point.y                              # 47.380989
xmin = utm_point.x                              # 8.5293235
xmax = utm_point.x + (m_1000_in_degree / 2)     # 8.548266
bb = hp_osm.BB(ymax=ymax, ymin=ymin, xmax=xmax, xmin=xmin)
gdf_bb = bb.as_gdf()
gdf_bb.crs = "epsg:{}".format(src_bb)
#gdf_bb_reprojected = gdf_bb.to_crs("epsg:{}".format(src_tif))
'''
ymin = y_topleft - 8000
ymax = y_topleft - 2000
xmin = x_topleft + 2000
xmax = x_topleft + 8000
bb = hp_osm.BB(ymax=ymax, ymin=ymin, xmax=xmax, xmin=xmin)
gdf_bb = bb.as_gdf(crs_orig=src_tif)

# OWN not saured clip geometry
gdf_bb = gpd.read_file(path_test_clip_polygon)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

# Get clip geometries
clip_geometries = getFeatures(gdf_bb)

# Create bounding box tiff
bb_tiff = hp_osm.BB(
    ymax=dataset.bounds.top,
    ymin=dataset.bounds.bottom,
    xmax=dataset.bounds.right,
    xmin=dataset.bounds.left)
gdf_bb_tiff = bb_tiff.as_gdf(src_tif)

#gdf_bb_tiff.to_file("C:/_scrap/SPARK/_tiffbb.shp")
#gdf_bb.to_file("C:/_scrap/SPARK/_bb.shp")

with rasterio.open(path_tif_attributed) as src:
    out_image, out_transform = rasterio.mask.mask(src, clip_geometries, crop=True)
    out_meta = src.meta
#tt = rasterio.open(path_tif,)
#out_image, out_transform = rasterio.mask.mask(dataset, clip_geometries, crop=True) 
#out_meta = src.meta
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform})
dataset = dataset.meta

with rasterio.open(path_tif_clipped, "w", **out_meta) as dest:
    dest.write(out_image)

# Clean temp files
os.remove(path_tif_attributed)

print("-----_finish_-----")
prnt("--")

'''# Orig
src = rasterio.open(path_tif)
array = src.read()
print("Array Shape: {}".format(array.shape))
print(src.crs)'''

# Read TIF with correct projection
#src = rasterio.open(path_tif_clipped)
#array = src.read()
#print("Array Shape: {}".format(array.shape))

# ------------------------
# Read with GDALL
# ------------------------
'''
dataset = gdal.Open(path_tif)#, gdal.GA_ReadOnly) 

print(dataset.GetProjection())
band = dataset.GetRasterBand(1)
arr = band.ReadAsArray()
plt.imshow(arr)
plt.show()

dataset = gdal.Open(path_tif)#, gdal.GA_ReadOnly) 
# Note GetRasterBand() takes band no. starting from 1 not 0
band = dataset.GetRasterBand(1)
arr = band.ReadAsArray()
#plt.imshow(arr)
#plt.show()
input_raster = gdal.Open(path_tif)
gdal.Warp(path_tif_clipped, input_raster, dstSRS='epsg:4326')'''
print("gdal finish")



'''
# -----------------
# Reproject TIF
# ------------------
with rasterio.Env():

    # As source: a 512 x 512 raster centered on 0 degrees E and 0
    # degrees N, each pixel covering 15".
    rows, cols = src_shape = (1000, 1000)
    d = 1.0 / 240 # decimal degrees per pixel
    # The following is equivalent to
    # A(d, 0, -cols*d/2, 0, -d, rows*d/2).
    src_transform = A.translation(-cols*d/2, rows*d/2) * A.scale(d, -d)
    src_crs = {'init': dst_crs}
    source = np.ones(src_shape, np.uint8)*255

    # Destination: a 1024 x 1024 dataset in Web Mercator (epsg:3857)
    # with origin at 0.0, 0.0.
    dst_shape = (1024, 1024)
    dst_transform = A.translation(-237481.5, 237536.4) * A.scale(425.0, -425.0)
    dst_crs = {'init': dst_crs}
    destination = np.zeros(dst_shape, np.uint8)

    reproject(
        source,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)

    # Assert that the destination is only partly filled.
    assert destination.any()
'''
#dataset = rasterio.open(path_tif)


'''ds = gdal.Open(path_tif)
prj = ds.GetProjection()
srs = osr.SpatialReference(wkt=prj)
if srs.IsProjected:
    print(srs.GetAttrValue('projcs'))'''
'''
new_dataset = rasterio.open(
    path_tif,
    'w',
    driver='GTiff',
    height=1000,
    width=1000,
    count=3,
    crs='+proj=latlong')'''


#print(src.transform)
with rasterio.open(path_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, 'epsg:{}'.format(src_tif), src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': 'epsg:{}'.format(src_tif),
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='epsg:{}'.format(src_tif),
                resampling=Resampling.nearest)
print("Finished")
prnt("fifn")


'''# WRITE OUT
new_dataset = rasterio.open(
    path_tif_attributed,
    'w',
    driver='GTiff',
    height=nr_of_cells,
    width=nr_of_cells,
    count=3,
    dtype='uint8',
    crs='epsg:{}'.format(src_tif),
    transform=transform)
new_dataset.write(dataset_no_attributed_data)'''


'''
# Open TIF with missing projection and add correct projection
with rasterio.open(
    path_tif_clipped,
    'w',
    driver='GTiff',
    height=1000,
    width=1000,
    count=3,
    dtype='uint8',
    crs='epsg:{}'.format(src_tif),
    transform=transform,
) as dst:
    print("----- Info -----")
    print(dst.crs)
    print(dst.transform)
    print(dst.bounds)
    print("---")
    array = dst.read()

    # Write TIF with correct projection
    ##dst.write(array)
'''