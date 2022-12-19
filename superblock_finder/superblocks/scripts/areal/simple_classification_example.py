"""Convert rasterio to shp to select easily raster cells

Taken from
http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html


"""
import os
import sys
import pprint
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, mapping
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.windows import Window
from rasterio.plot import reshape_as_raster, reshape_as_image

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Paths
path_tif = "K:/superblocks/01-data/assing_labels_to_tif/small_test.tif"
path_manual_classification = "K:/superblocks/01-data/assing_labels_to_tif/manual_classification.shp"
path_out = "K:/superblocks/01-data/assing_labels_to_tif"

# Name of attribute of manual classification
class_attribute = 'labels'

da = xr.open_rasterio(path_tif)

# Convert tif to gdf rasterfile, # Note: Needs tif with correctly attributed metadat
tif_dataset = rasterio.open(path_tif)
print("----Metadata tiff---")
pprint.pprint(tif_dataset.meta)

# Get coordinates
nr_of_rows = tif_dataset.shape[0]     # Hight
nr_of_columns = tif_dataset.shape[1]  # Width (Breite)

# Coordinates
x_list, y_list = tif_dataset.xy(range(nr_of_rows), range(nr_of_columns))


'''
# CREATE Point Shapefile
coordinate_grid_list = np.meshgrid(np.array(x_list), np.array(y_list))
print("coordinate_grid: {}".format(coordinate_grid_list))
# Create point gdf (very data hungry)
print("... creating point shapefile")
coords = []
for i in x_list:
    for j in y_list:
        coords.append(Point(i, j))
gdf_grid = gpd.GeoDataFrame(geometry=coords)
gdf_grid.to_file(os.path.join(path_out, "grid_centroids.shp"))
print("finished creating point shapefile")'''

# -------------------------------
# Get manually labeled geometries
# -------------------------------
manual_shps = gpd.read_file(path_manual_classification)
band_count = tif_dataset.count  # Get number of bands of image

# Specify dtype of values and create values and labels
y = np.array([], dtype=np.string_)  # labels for training
X = np.array([], dtype=np.int).reshape(0, band_count)  # pixel values for training

# Iterate geometries with labels and stick all labels together
for index in manual_shps.index:
    label = manual_shps.loc[index][class_attribute]
    geometry = manual_shps.loc[index]['geometry']
    feature = [mapping(geometry)]
    try:
        out_image, out_transform = mask(tif_dataset, feature, crop=True)
    except:
        raise Exception("Could not get labels")

    # Eliminate all the pixels with 0 values and 255 for all bands (not actually part of the shapefile)
    out_image_trimmed = out_image[:, ~np.all(out_image == 0, axis=0)]
    out_image_trimmed = out_image_trimmed[:, ~np.all(out_image_trimmed == 255, axis=0)]

    # reshape the array to [pixel count, bands]
    out_image_reshaped = out_image_trimmed.reshape(-1, band_count)
    # append the labels to the y array
    nr_of_pixels = out_image_reshaped.shape[0]
    y = np.append(y, [manual_shps[class_attribute][index]] * nr_of_pixels)  #nr_of_bands
    # stack the pizels onto the pixel array
    X = np.vstack((X, out_image_reshaped))      
    print("Collecting labels from geometry: Label: {} Size: {}".format(label, out_image.shape))


# Get all classification labels
labels = np.unique(manual_shps[class_attribute])
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, classes=labels))

# We will need a "X" matrix containing our features, and a "y" array containing our labels
print('Our X matrix is sized: {}'.format(X.shape))
print('Our y array is sized: {}'.format(y.shape))


# Plot mean pixel values across all bands for each label to see how they segment
fig, ax = plt.subplots(1, 1)

bands = np.arange(1, band_count + 1) # number of bands
classes = np.unique(y)

for class_type in classes:
    all_values_with_class_type = X[y == class_type, :]
    band_intensity = np.mean(all_values_with_class_type, axis=0)
    print("INno: {}  {}".format(band_intensity, class_type))
    ax.plot(bands, band_intensity, label=class_type)

ax.set_xlabel('Band #')
ax.set_ylabel('Reflectance Value')
ax.legend(loc="upper right")
ax.set_title('Band Intensities Full Overview')
#plt.show()


# --------TRAINING CLASSIFICATION with sklearn
#classifier = GaussianNB()
classifier = RandomForestClassifier()
classifier.fit(X, y)


with rasterio.open(path_tif) as src:
    # may need to reduce this image size if your kernel crashes, takes a lot of memory
    img = src.read()

    # Get all pixels which have not the defined value across all dimensions
    value_to_filter = 0  #and 255: TODO:
    grid_to_assign_borderpixels = ~np.all(img == value_to_filter, axis=0)
    #img_trimmed = img[:, np.all(img == value_to_filter, axis=0)]
    # Replace value


# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
reshaped_img = reshape_as_image(img)

# Predicts
class_prediction = classifier.predict(reshaped_img.reshape(-1, band_count))

# Reshape our classification map back into a 2D matrix so we can visualize it
class_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)


def str_class_to_int(class_array):
    """Convert string to int"""
    class_array[class_array == 'urban'] = 1
    class_array[class_array == 'rural'] = 2
    return(class_array.astype(int))


# Convert string labels to int labels
class_prediction = str_class_to_int(class_prediction)

# Replace boundary pixes with 0 values
class_prediction = class_prediction * grid_to_assign_borderpixels

# Relabel all outside pixels
#TODO:

# find the highest pixel value in the prediction image
n = int(np.max(class_prediction))

# next setup a colormap for our map
rgb_colors = dict((
    (0, (0, 0, 0, 0)),          # Border pixel
    (1, (139, 69, 19, 255)),    # Brown - Urban
    (2, (34, 139, 34, 255)),    # Green - Rural

    
))

# Put 0 - 255 as float 0 - 1
for k in rgb_colors:
    v = rgb_colors[k]
    _v = [_v / 255.0 for _v in v]
    rgb_colors[k] = _v

index_colors = [rgb_colors[key] if key in rgb_colors else (255, 255, 255, 0) for key in range(0, n+1)]

cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n+1)

# Plotting classified and original image
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

def color_stretch(image, index):
    """ need to understand better"""
    colors = image[:, :, index].astype(np.float64)
    nr_of_bands = colors.shape[2]
    for band in range(nr_of_bands):
        colors[:, :, band] = rasterio.plot.adjust_band(colors[:, :, band])
    return colors

img_stretched = color_stretch(reshaped_img, [0, 1, 2])

axs[0].imshow(img_stretched)                                        # Original image
axs[1].imshow(class_prediction, cmap=cmap, interpolation='none')    # Classified image


# Create Legend
legend_handles = [
    mpatches.Patch(color=rgb_colors[0], label="Outside"),
    mpatches.Patch(color=rgb_colors[1], label="Urban"),
    mpatches.Patch(color=rgb_colors[2], label="Rural")]

legend = plt.legend(
    handles=legend_handles,
    prop={'size': 8},
    loc='upper right',
    #bbox_to_anchor=(0.5, -0.05),
    frameon=True)

plt.show()
print("--finished--")
