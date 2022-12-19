"""

LV95
https://shop.swisstopo.admin.ch/de/products/images/ortho_images/SWISSIMAGE_RS

Bodenauflösung : 0.1 m im Flachland, 0.25 m in Berggebieten

Spektrale Auflösung : NIR [808 - 882 nm], Rot [619 - 651 nm], Grün [525 – 585 nm] und Blau [435 – 495 nm]

Farbtiefe : 16 Bit ohne Histogrammstreckung

Format : TIFF mit Komprimierung ZIP + TFW (World File) – Georeferenz

Koordinatensystem : LV95 --> 2056

Nachführung : SWISSIMAGE RS ist ab den Bildaufnahmen 2016 verfügbar und deckt die gesamte Fläche der Schweiz und Liechtenstein ab. Die Nachführung erfolgt in einem Dreijahreszyklus.

Struktur
Bei SWISSIMAGE RS werden jeweils Teilstücke von orthorektifizierten Bildstreifen ausgeliefert. Jeder Bildstreifen besitzt eine Identifikationsnummer, die einen Rückschluss auf das Datum und die Zeit der Bildaufnahme zulässt. Alle Teilstücke eines Bildstreifens enthalten denselben Identifikator, gefolgt von einer Zusatznummer (0_0, 0_1, 0_2,…), damit diese unterschieden werden können.

Beispiel : Die Teilstücke 20160420_1253_30030_0_4 und 20160420_1253_30030_0_5 gehören zum Bildstreifen 20160420_1253_30030. Die ersten 8 Ziffern leiten sich vom Datum, die 4 darauffolgenden vom Zeitpunkt (GMT+0) der Bildaufnahme ab. In diesem Beispiel handelt es sich um einen Bildstreifen, der am 20.04.2016 um 12:53 Uhr (GMT+0) aufgenommen wurde. Um den Aufnahmezeitpunkt für die Schweizer Zeitzone zu erhalten, müssen während der Winterzeit eine Stunde (GMT+1) und während der Sommerzeit zwei Stunden (GMT+2) hinzugefügt werden.

Im Gegensatz zu SWISSIMAGE (RGB) wird aus den Einzelorthofotos kein Mosaik erstellt. Entsprechend ist kein homogenes Erscheinungsbild über eine grosse Fläche möglich. Ausserdem kommt es bei den verschiedenen Bildstreifen zu Bildüberlappungen.

Qualität
+/- 0.15 m (1 sigma) für die Bodenauflösung 0.1 m und +/- 0.25 m (1 sigma) für die Bodenauflösung 0.25 m, +/- 3-5 m (1 sigma) in unebenem Gelände.

Diese Qualitätsmerkmale gelten für den, in Flugrichtung zentralen Bereich eines Orthobildstreifens. Am Streifenrand kann die planimetrische Genauigkeit abnehmen. Diese Genauigkeit ist schwierig zu bestimmen und wird hauptsächlich von der Qualität des digitalen Geländemodells, welches für die Orthorektifizierung benötigt wird, beeinflusst.

Im Gegensatz zu SWISSIMAGE (RGB) werden keine radiometrischen und geometrischen Anpassungen vorgenommen.

Height: 22165
Widht: 10240
Resolution: 10 cm
"""
resolution = 0.1  # [m]
crs_swisss_image = 2056
#NOTE: IF OPENED IN epsg:2056, correct
import os
import sys
import pprint
import rasterio

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import helper_osm as hp_osm

path_meta = "C:/____TRA/20190323_1142_12501_0_57.tfw"
path_tif = "C:/____TRA/20190323_1142_12501_0_57.tif"
path_tif_clipped = "C:/____TRA/REFERENCED.tif"

tfw_file = open(path_meta, "r")
tfw_raw_data = tfw_file.read()  #reading all text from file

# Read metadata file (affine transformation)
tfw_raw_data_splitted = []
for i in tfw_raw_data.split("\n"):
    if i != "":
        tfw_raw_data_splitted.append(float(i))
pprint.pprint(tfw_raw_data_splitted)

x_top_left = tfw_raw_data_splitted[4]
y_top_left = tfw_raw_data_splitted[5]
print("Top left x coordinate: {}".format(x_top_left))
print("Top left y coordinate: {}".format(y_top_left))


# ---- Reading actual file
swissimage_rs = rasterio.open(path_tif)


print("METADATA")
pprint.pprint(swissimage_rs.meta)

print("Height: {}".format(swissimage_rs.meta['height']))
print("width: {}".format(swissimage_rs.meta['width']))
      
x_top_left = swissimage_rs.meta['transform'][2]
y_top_left = swissimage_rs.meta['transform'][5]
x_bottom_right = x_top_left + (swissimage_rs.meta['width'] * resolution)
y_bottom_right = y_top_left - (swissimage_rs.meta['height'] * resolution)


bb = hp_osm.BB(ymax=y_top_left, ymin=y_bottom_right, xmax=x_bottom_right, xmin=x_top_left)
bb_gdf = bb.as_gdf(crs_orig=crs_swisss_image)
bb_gdf.to_file("C:/_scrap/A.shp")

print("Top left coordinates: {}  {}".format(x_top_left, y_top_left))
print("bottom right coordinates: {}  {}".format(x_bottom_right, y_bottom_right))

out_meta = swissimage_rs.meta
out_meta['crs'] = 2056 # ORIGINAL SWISSIMAGE RS


#"height": out_image.shape[1],
#"width": out_image.shape[2],
#"transform": out_transform})
swissimage_rs = swissimage_rs.meta

with rasterio.open(path_tif_clipped, "w", **out_meta) as dest:
    dest.write(swissimage_rs)


#with rasterio.open(path_tif) as src:
#    out_image, out_transform = rasterio.mask.mask(src, clip_geometries, crop=True)
#    out_meta = src.meta
    
    
print("___F____")