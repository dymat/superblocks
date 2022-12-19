"""Iterate files in folder and plot statistics
"""
import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import helper_read_write as hp_rw


path_folder = "K:/superblocks/02-superblock_scenario/zurich/blocks/areas/superblock"
#path_folder = "K:/superblocks/02-superblock_scenario/zurich/blocks/areas/miniblock"
path_folder = "K:/superblocks/02-superblock_scenario/zurich/blocks/areas/miniblockS"
all_files_destination = os.listdir(path_folder)

ha_factor = 10000
result_conatiner = []

for file_name in all_files_destination:
    if file_name.endswith('.shp'):
        id_superblock = int(file_name.split("__")[1][:-4])
        gpd_raw = gpd.read_file(os.path.join(path_folder, file_name))

        area = sum(list(gpd_raw.area)) / ha_factor  #   convert m to ha

        result_conatiner.append(["id_{}".format(id_superblock), area])


fig, ax = plt.subplots(figsize=hp_rw.cm2inch(9, 9))

df_raw = pd.DataFrame(result_conatiner, columns=['id_superblock', 'area'])
df_raw = df_raw.set_index('id_superblock')
df_raw = df_raw.sort_values(by=['area'], ascending=False)
df_raw.plot(y='area', kind='bar', color='grey', ax=ax)


plt.title("MiniSblock statistics (street area)")
plt.xlabel("MiniSblock id", labelpad=10, weight='bold', size=8)
plt.ylabel("Area (ha)", labelpad=10, weight='bold', size=8)  #$km^{2}$)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()