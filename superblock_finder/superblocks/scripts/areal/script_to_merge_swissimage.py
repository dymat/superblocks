"""Script to get correct swissimage file based on polygon to clip


"""
import os
import sys
import geopandas as gpd
import rasterio.merge

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.areal import vhm_helper as vhm_hp

src_tif = 2056
target_crs = 32632


#path_clip = "C:/_results_superblock/zurich/blocks/areas/superblock/superblock__18.shp"
path_clip = "K:/superblocks/01-data/cities/zurich/extent.shp"
#path_clip = "K:/superblocks/01-data/cities/zurich/extent_full.shp"
image_path = "T:/zh"
path_out = "T:/zh_merged"
path_out_merged = os.path.join(path_out, "merged_notransform.tif")
extents_path = os.path.join(image_path, "extents/extents.shp")
path_reprojected = os.path.join(path_out, "merged_{}.tif".format(target_crs))


# read clip polygon
clip_polygon = gpd.read_file(path_clip)

# Read extents
extents = gpd.read_file(extents_path)

# Transform extents
extents = extents.to_crs(target_crs)
print("Clip crs: {}".format(clip_polygon.crs))
print("extents crs: {}".format(extents.crs))

assert clip_polygon.crs == extents.crs
# Get all images which need to be merged
images_to_merge = []
for extent_loc in extents.index:
    extent_name = extents.loc[extent_loc]['names']
    geometry_extent = extents.loc[extent_loc].geometry
    image_name = os.path.join(image_path, extent_name.replace(".shp", ".tif"))

    with rasterio.open(image_name) as src:
        print("_-")
        
    # Envelop or minimum_rotated_rectangle
    if geometry_extent.envelope.contains(clip_polygon.envelope[0]) or geometry_extent.envelope.intersects(clip_polygon.envelope[0]):
        images_to_merge.append(image_name)
    #images_to_merge.append(image_name)

print("Name of tif images to merge")
print(images_to_merge)
print(len(images_to_merge))
images_to_merge = [images_to_merge[0]]
if images_to_merge == []:
    raise Exception("No Tifs found to merge")

# Merge image and save as image (https://gist.github.com/nishadhka/9bc758129c2949a3194b79570198f544)
rasterio_all_datasets = []
for image_to_merge in images_to_merge:
    #print("merge iteratively: {}".format(image_to_merge))
    data_file = rasterio.open(image_to_merge)
    rasterio_all_datasets.append(data_file)
    #dest, output_transform = rasterio.merge.merge(rasterio_all_datasets, bounds=None, res=None, nodata=None)

dest, output_transform = rasterio.merge.merge(rasterio_all_datasets, bounds=None, res=None, nodata=None)

with rasterio.open(images_to_merge[0]) as src:
    out_meta = src.meta.copy()    
    out_meta.update(
        {"driver": "GTiff",
        "crs": 'epsg:{}'.format(src_tif),  # Assign orig projection crs='epsg:{}'.format(src_tif),
        "height": dest.shape[1], 
        "width": dest.shape[2],
        "transform": output_transform})

with rasterio.open(path_out_merged, "w", **out_meta) as dest1:
    dest1.write(dest)

# Reproject
vhm_hp.rasterio_reproject_function(
    path_tif_from=path_out_merged,
    path_reprojected=path_reprojected,
    src_target=target_crs)

print("finish")