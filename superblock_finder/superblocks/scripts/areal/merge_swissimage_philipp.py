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
    
    # Envelop or minimum_rotated_rectangle
    if geometry_extent.envelope.contains(clip_polygon.envelope[0]) or geometry_extent.envelope.intersects(clip_polygon.envelope[0]):
        images_to_merge.append(image_name)
    #images_to_merge.append(image_name)

print("Name of tif images to merge")
print(images_to_merge)
print(len(images_to_merge))
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
        "transform": output_transform
        })

with rasterio.open(
    path_out_merged, "w", **out_meta) as dest1:
    dest1.write(dest)

# Reproject
vhm_hp.rasterio_reproject_function(
    path_tif_from=path_out_merged,
    path_reprojected=path_reprojected,
    src_target=target_crs)

print("finish")


def get_correct_10cm_image(
        clip_polygon,
        path_extents,
        path_source_gifs,
        path_temp_gifs,
        target_crs=32632
    ):
    """
    Check if clip polygon is fully contained within a polygon
    If yes, then reproject to target crs and provide out path

    If intersects multiple files, merge them and project them.
    Return path to this mergd projected fil

    Inputs
    -------
    clip_polygon        :   Geopandas of clip polygon
    path_extents        :   Path to extent shapefiles of swissimage files
    path_temp_gifs      :   Folder temp path to save collated geotifs

    Return
    ------
    path to projected geotif which fully contains the clip geometry
    """
    # Read extents
    extents = gpd.read_file(path_extents)
    extents = extents.to_crs(target_crs)

    # Check that both same crs
    assert clip_polygon.crs == extents.crs

    # Check if clip polyon fully contained within a single tif
    imag_contained = []
    imag_intersected = []
    for extent_loc in extents.index:
        extent_name = extents.loc[extent_loc]['names']
        geometry_extent = extents.loc[extent_loc].geometry
        image_name = os.path.join(path_source_gifs, extent_name.replace(".shp", ".tif"))

        # Envelop or minimum_rotated_rectangle
        if geometry_extent.envelope.contains(clip_polygon.envelope[0]):
            imag_contained.append(image_name)
        if geometry_extent.envelope.intersects(clip_polygon.envelope[0]):
            imag_intersected.append(image_name)

    # If contains
    if len(imag_contained) > 0:
        first_containing_image = imag_contained[0]

        path_matching_tif = os.path.join(path_source_gifs, first_containing_image)
        path_reprojected_tif = os.path.join(path_temp_gifs, first_containing_image)

        # If file not yet exists
        if not os.path.exists(path_reprojected_tif):
            # Reproject
            vhm_hp.rasterio_reproject_function(
                path_tif_from=path_matching_tif,
                path_reprojected=path_reprojected_tif,
                src_target=target_crs)
            path_to_choose = path_reprojected_tif
        else:
            print("already written file")
            path_to_choose = path_reprojected_tif

    elif len(imag_intersected) > 1:
        # Paths
        projected_name = "projected_{}_{}.tif".format(
            clip_polygon.envelope[0].centroid.x,
            clip_polygon.envelope[0].centroid.y)
        path_reprojected_tif = os.path.join(path_temp_gifs, projected_name)

        if not os.path.exsts(path_reprojected_tif):
            # IF within multiple images, make a merge
            dest, output_transform = rasterio.merge.merge(imag_intersected, bounds=None, res=None, nodata=None)

            with rasterio.open(imag_intersected[0]) as src:
                out_meta = src.meta.copy()    
                out_meta.update({
                    "driver": "GTiff", "crs": 'epsg:{}'.format(src_tif), "height": dest.shape[1], 
                    "width": dest.shape[2], "transform": output_transform})

            with rasterio.open(path_reprojected_tif, "w", **out_meta) as dest1:
                dest1.write(dest)
            path_to_choose = path_reprojected_tif
        else:
            print("... arelads projected")
            path_to_choose = path_reprojected_tif
    else:
        raise Exception("No matching image")

    return path_to_choose