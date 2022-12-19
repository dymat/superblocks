"""Extract swissimage

Info
-----
SwissIMage: epsg:2056: CH1903+_LV95
"""
import pandas as pd
from shapely.geometry import Polygon, Point

path_metadata = "K:/superblocks/01-data/_image/DOP10_LV95_Actual.csv"
path_metadata = "K:/superblocks/01-data/_image/_up_nov_17_DOP25_LV03_Actual.csv"


# Extraction bounding box
ymin = 47.3601065
ymax = 47.380989
xmin = 8.5293235
xmax = 8.548266

coords = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
bb_extraction = Polygon(coords)

# fields
metadata = pd.read_csv(path_metadata, delimiter=';')

# Gett all path of images which intersect the extraction bouding box
images_paths = []

for index in metadata.index:
    resolution = metadata.loc[index]['ResolutionOfOrigin']
    xmin = metadata.loc[index]['UpperLeftX']
    ymax = metadata.loc[index]['UpperLeftY']
    crs_origin = metadata.loc[index]['SourceReferenceSystem']
    # Calculate lowerRight Coordinate
    ymax = xmin + 1000  # [m]
    ymin = ymax - 1000  # [m]

    # Create polygon of swissimage bb
    coords = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
    bb_image = Polygon(coords)

    # Tesf it point in 
    p1 = Point(2682919.7, 1246815.4)
    if bb_image.contains(p1):
        print("JJJJJJJ")
    # Test if intersect
    if bb_image.intersects(bb_extraction):
        file_name = metadata.loc[index]['Filename']
        images_paths.append(file_name)
    if bb_image.contains(bb_extraction):
        print("fullywtithing")
        images_paths.append(file_name)
    
print(images_paths)

print("___finish___")