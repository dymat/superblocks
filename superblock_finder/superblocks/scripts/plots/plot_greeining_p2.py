"""Plots of urban greening for report
"""
import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import helper_plot as hp_plt
from superblocks.scripts.areal import rasterio_helper as hp_rio
from superblocks.scripts.network import helper_network as hp_net


label_dict = hp_rw.city_labels_ch()  #load labels

# -------------------------------------------------------------------------
print("NOTE: this screep first needs to run /areal/segmentation_urban_greening.py")
# -------------------------------------------------------------------------

# Local
path_ch_urban_green = "C:/DATA/ginzler_greening/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95.tif"
path_GWR_out_segment = "C:/_results_swiss_communities"
path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"

# Server
##path_ch_urban_green = "J:\Sven\_DATA\ginzler_greening/Vegetationshoehenmodell_2019_1m_LFI_ID164_19_LV95.tif"
##path_GWR_out_segment = "J:/Sven/greening/_results_swiss_communities"
##path_communities = "J:/Sven/_DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"

path_out = "H:/09-papers/2021-superblocks_urban_greening/_results_simulation"

gemeinden = gpd.read_file(path_communities)
gemeinden = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren'])]
segmentation_IDs = list(gemeinden['BFS_NO'])
segmentation_IDs.sort()

hp_rw.create_folder(path_out)

ha_factor = 10000

stretch_factor = "0.5"
case_studies = [230, 261, 351, 1061, 2701, 3203, 5192, 5586, 6621]
path_results = "C:/_results_swiss_communities/_results/"
path_buildings = "C:/DATA/buildings_superblock_greening/buildings_major_cities.shp"

buildings = gpd.read_file(path_buildings)

# Figure collectors
collector_fig_green_tot_block = []
collector_fig_green_tot_street = []
collector_fig_far_green = pd.DataFrame()
cnter = 0
for segmentation_id in case_studies:
    print("collecting data...{}".format(segmentation_id))
    cnter += 1000

    # Get label
    gemeinden_label = list(gemeinden.loc[gemeinden['BFS_NO'] == segmentation_id]['GDE_NAME'])[0]

    # Get building footprint coverage
    buildings_bfs = buildings.loc[buildings['BFS_NR'] == segmentation_id]
    buildings_bfs = buildings_bfs.reset_index()
    buildings_bfs = buildings_bfs.to_crs(32632)

    # Create search tree
    rtree_buildings = hp_net.build_rTree(buildings_bfs)

    # Get block and street geometry
    block_path = os.path.join(path_results, str(segmentation_id), 'blocks', str(stretch_factor), "block_all.shp")
    blocks = gpd.read_file(block_path)
    streets_path = os.path.join(path_results, str(segmentation_id), 'blocks', str(stretch_factor), "street_all.shp")
    streets = gpd.read_file(streets_path)

    streets['geometry'] = streets.geometry.buffer(0.1).buffer(-0.2).buffer(0.1)
    streets = streets[~streets.is_empty]
    streets = streets[streets.is_valid]

    assert blocks.crs == buildings_bfs.crs

    # Iterate all individual blocks
    path_green_files = os.path.join(path_results, str(segmentation_id), 'blocks', str(stretch_factor), 'green')
    all_ids = os.listdir(path_green_files)

    collector = []

    for path_id in all_ids:
        id_block = int(path_id.split("_")[-1][:-4])
        id_block_out = id_block + cnter
        type_block = path_id.split("_")[1]
        flora = 0

        # Of block (and not street)
        block_geometry = list(blocks.loc[blocks['inter_id'] == id_block].geometry)[0]
        sub_category = list(blocks.loc[blocks['inter_id'] == id_block]['b_type'])[0]
        block_area = block_geometry.area

        # Get buildings in block
        potential_intersections = list(rtree_buildings.intersection(block_geometry.bounds))
        # Interate cadastre and clip if intersection
        for potential_intersection in potential_intersections:
            cadastre_geom = buildings_bfs.loc[potential_intersection].geometry
            if block_geometry.contains(cadastre_geom):
                flora += buildings_bfs.loc[potential_intersection]['flora']

        # Calculate far
        far = flora / block_area

        if type_block == 'street':
            street_area = list(streets.loc[streets['inter_id'] == id_block].geometry.area)[0]
            block_area = street_area

        src = rasterio.open(os.path.join(path_green_files, path_id))
        array = src.read()

        # Classify green
        green_threshold = 0.05  # Threshold
        area_pixel = 1  # [m2]
        n = len(array[array >= green_threshold])

        # Calculate area
        green_m2 = n * area_pixel
        non_green_m2 = round(block_area, 2) - green_m2

        if non_green_m2 < 0:
            print("_--")
        assert green_m2 >= 0
        assert non_green_m2 > 0

        collector.append([gemeinden_label, id_block_out, type_block, sub_category, block_area, green_m2, non_green_m2, flora, far])

    # Result
    pd_result = pd.DataFrame(collector, columns=['name', 'id', 'type', 'sub_cat', 'tot', 'green', 'nongreen', 'flora', 'far'])

    # -------
    # Summary
    # -------
    gdf_blocks = pd_result[pd_result['type'] == 'block']
    gdf_street = pd_result[pd_result['type'] == 'street']

    tot_area_block = gdf_blocks['tot'].sum() / ha_factor
    tot_non_green_block = gdf_blocks['nongreen'].sum() / ha_factor
    tot_green_block = gdf_blocks['green'].sum() / ha_factor

    tot_area_street = gdf_street['tot'].sum() / ha_factor
    tot_non_green_street = gdf_street['nongreen'].sum() / ha_factor
    tot_green_street = gdf_street['green'].sum() / ha_factor

    # Street to block factor
    f_street_block = tot_area_street / tot_area_block

    # Add sumary results
    collector_fig_green_tot_block.append([gemeinden_label, tot_non_green_block, tot_green_block])
    collector_fig_green_tot_street.append([gemeinden_label, tot_non_green_street, tot_green_street])

    # Collector global
    collector_fig_far_green = collector_fig_far_green.append(pd_result, ignore_index=True)

colors_labels = ['#a7c957', '#a0acb2']  ##386641  '#ee9b00 ['#848f4b', '#a0acb2']

collector_fig_far_green = collector_fig_far_green.rename(index=label_dict)  #Rename Index

#NOTE: SOMETHING WITH THE TOTAL BLOCK AREA IS NOT WORKING
# ----------------------------------------------
# Figure - FAR vers greening in BLOCK for all case studies
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))

# Calculate percent green
block_all_info_block = collector_fig_far_green[collector_fig_far_green['type'] == 'block']
block_all_info_block['p_green'] = (block_all_info_block['green'] / block_all_info_block['tot']) * 100

for sub_cat, color in zip(set(block_all_info_block['sub_cat'].values.tolist()), colors_labels):
    block_all_info_cat = block_all_info_block.loc[block_all_info_block['sub_cat'] == sub_cat]
    size = list(int(i) / 1000 for i in block_all_info_cat['tot'])
    block_all_info_cat = block_all_info_cat.set_index('type')
    block_all_info_cat.plot(
        kind='scatter',
        stacked=True,
        ax=ax,
        x='far',
        y='p_green',
        s=size,
        color=color,
        label=sub_cat,
        alpha=0.8,
        edgecolor='black',
        linewidth=0)

plt.xlabel("floor area ratio", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.ylabel("urban green space in block (%)", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')

leg = plt.legend()

leg.get_frame().set_linewidth(0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.get_legend().remove()

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'block_far_pgreen.pdf'))
block_all_info_block.to_csv(os.path.join(path_out, 'block_far_pgreen.csv'))
# Plot street categorization as in publication

# ----------------------------------------------
# Figure - FAR vers greening in STREET for all case studies
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))

# Calculate percent green
collector_fig_far_green['p_green'] = (collector_fig_far_green['green'] / collector_fig_far_green['tot']) * 100
block_all_info_street = collector_fig_far_green[collector_fig_far_green['type'] == 'street']

# Add FAR to streets
for i in block_all_info_street.index:
    id_street = block_all_info_street.loc[i]['id']
    entry_block = collector_fig_far_green.loc[(collector_fig_far_green['id'] == id_street) & (collector_fig_far_green['type'] == 'block')]
    block_all_info_street.at[i, 'far'] = entry_block['far'].values.tolist()[0]

for sub_cat, color in zip(set(block_all_info_street['sub_cat'].values.tolist()), colors_labels):
    block_all_info_cat = block_all_info_street.loc[block_all_info_street['sub_cat'] == sub_cat]
    size = list(int(i) / 500 for i in block_all_info_cat['tot'])
    block_all_info_cat = block_all_info_cat.set_index('type')
    block_all_info_cat.plot(
        kind='scatter',
        stacked=True,
        ax=ax,
        x='far',
        y='p_green',
        color=color,
        s=size,
        label=sub_cat)

plt.xlabel("floor area ratio", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.ylabel("green street area (%)", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')
ax.ticklabel_format(style='plain')  # deactivate scientific notation

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.get_legend().remove()
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'street_far_pgreen.pdf'))
block_all_info_street.to_csv(os.path.join(path_out, 'street_far_pgreen.csv'))

# ----------------------------------------------
# Figure (STREET) Stacked greening vs non greening (absolute)
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))
pd_fig1 = pd.DataFrame(collector_fig_green_tot_street, columns=['name', 'grey', 'green'])
pd_fig1 = pd_fig1[['name', 'green', 'grey']]
pd_fig1 = pd_fig1.set_index('name')
pd_fig1 = pd_fig1.rename(index=label_dict)  #Rename Index

# Sort according to grey size
pd_fig1 = pd_fig1.sort_values(by=['grey'], ascending=False)

pd_fig1.plot.barh(stacked=True, ax=ax, width=0.7, color=colors_labels, alpha=0.5)

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
ax.get_legend().remove()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Add percentage as label and# calculate absolte values per city
cities_to_calc = pd_fig1.index.tolist()
abs_vals = {}
cnt = 0
for i in range(2):  # green and non-green
    for city_to_calc in cities_to_calc:
        abs_vals[cnt] = pd_fig1.loc[city_to_calc].values.sum()
        cnt += 1

for cnt, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    percentage = (100 / abs_vals[cnt]) * width # Note: if bar: height  for barh: widht
    ax.text(
        x + width / 2, 
        y + height / 2, 
        '{:.0f}'.format(percentage), 
        horizontalalignment='center', 
        verticalalignment='center',
        fontname='Arial',
        size=6)

plt.ylabel(" ", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xlabel("street area (ha)", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')
#ax.ticklabel_format(style='plain')  # deactivate scientific notation

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'total_street.pdf'))
pd_fig1.to_csv(os.path.join(path_out, 'total_street.csv'))


# ----------------------------------------------
# Figure (STREET) Stacked greening vs non greening (relative)
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))
pd_fig1 = pd.DataFrame(collector_fig_green_tot_street, columns=['name', 'grey', 'green'])

pd_fig1 = pd_fig1[['name', 'green', 'grey']]
pd_fig1 = pd_fig1.set_index('name')
pd_fig1 = pd_fig1.rename(index=label_dict)  #Rename Index

# Divide by sum of row,c alculate relative area
pd_fig1 = pd_fig1.div(pd_fig1.sum(axis=1), axis=0) * 100

# Sort according to grey size
pd_fig1 = pd_fig1.sort_values(by=['grey'], ascending=False)

pd_fig1.plot.barh(stacked=True, ax=ax, width=0.7, color=colors_labels, alpha=0.5)

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
ax.get_legend().remove()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Add percentage as label and# calculate absolte values per city
cities_to_calc = pd_fig1.index.tolist()
abs_vals = {}
cnt = 0
for i in range(2):  # green and non-green
    for city_to_calc in cities_to_calc:
        abs_vals[cnt] = pd_fig1.loc[city_to_calc].values.sum()
        cnt += 1

for cnt, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    percentage = (100 / abs_vals[cnt]) * width # Note: if bar: height  for barh: widht
    ax.text(
        x + width / 2, 
        y + height / 2, 
        '{:.0f}'.format(percentage), 
        horizontalalignment='center', 
        verticalalignment='center',
        fontname='Arial',
        size=6)

plt.xlim(0, 100)
plt.ylabel(" ", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xlabel("street area (ha)", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')
#ax.ticklabel_format(style='plain')  # deactivate scientific notation

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'total_street_relative.pdf'))
pd_fig1.to_csv(os.path.join(path_out, 'total_street_relative.csv'))



# ----------------------------------------------
# Figure
# Stacked greening vs non greening (absolute)
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))
pd_fig1 = pd.DataFrame(collector_fig_green_tot_block, columns=['name', 'grey', 'green'])
pd_fig1 = pd_fig1[['name', 'green', 'grey']]
pd_fig1 = pd_fig1.set_index('name')
pd_fig1 = pd_fig1.rename(index=label_dict)  #Rename Index

# Sort according to grey size
pd_fig1 = pd_fig1.sort_values(by=['grey'], ascending=False)

pd_fig1.plot.barh(stacked=True, ax=ax, width=0.7, color=colors_labels, alpha=0.5)

# Add percentage as label and# calculate absolte values per city
cities_to_calc = pd_fig1.index.tolist()
abs_vals = {}
cnt = 0
for i in range(2):  # green and non-green
    for city_to_calc in cities_to_calc:
        abs_vals[cnt] = pd_fig1.loc[city_to_calc].values.sum()
        cnt += 1

for cnt, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    percentage = (100 / abs_vals[cnt]) * width  # Note: if bar: height  for barh: widht
    ax.text(
        x + width / 2, 
        y + height / 2, 
        '{:.0f}'.format(percentage), 
        horizontalalignment='center', 
        verticalalignment='center',
        fontname='Arial',
        size=6)

plt.ylabel(" ", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xlabel("block area (ha)", fontname='Arial', labelpad=5, weight='bold', size=10) #$\mathdefault{ha}$)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')
#ax.ticklabel_format(style='plain')  # deactivate scientific notation
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'total_block.pdf'))
pd_fig1.to_csv(os.path.join(path_out, 'total_block.csv'))


# ----------------------------------------------
# Figure
# Stacked greening vs non greening (relative)
# ----------------------------------------------
fig, ax = plt.subplots(figsize=hp_plt.cm2inch(9, 9))
pd_fig1 = pd.DataFrame(collector_fig_green_tot_block, columns=['name', 'grey', 'green'])
pd_fig1 = pd_fig1[['name', 'green', 'grey']]
pd_fig1 = pd_fig1.set_index('name')
pd_fig1 = pd_fig1.rename(index=label_dict)  #Rename Index

# Divide by sum of row, calculate relative area
pd_fig1 = pd_fig1.div(pd_fig1.sum(axis=1), axis=0) * 100

# Sort according to grey size
pd_fig1 = pd_fig1.sort_values(by=['grey'], ascending=False)

pd_fig1.plot.barh(stacked=True, ax=ax, width=0.7, color=colors_labels, alpha=0.5)

# Add percentage as label and# calculate absolte values per city
cities_to_calc = pd_fig1.index.tolist()
abs_vals = {}
cnt = 0
for i in range(2):  # green and non-green
    for city_to_calc in cities_to_calc:
        abs_vals[cnt] = pd_fig1.loc[city_to_calc].values.sum()
        cnt += 1

for cnt, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    percentage = (100 / abs_vals[cnt]) * width # Note: if bar: height  for barh: widht
    ax.text(
        x + width / 2, 
        y + height / 2, 
        '{:.0f}'.format(percentage), 
        horizontalalignment='center', 
        verticalalignment='center',
        fontname='Arial',
        size=6)
plt.xlim(0, 100)
plt.ylabel(" ", fontname='Arial', labelpad=5, weight='bold', size=10)
plt.xlabel("block area (ha)", fontname='Arial', labelpad=5, weight='bold', size=10) #$\mathdefault{ha}$)
plt.xticks(fontsize=8, fontname='Arial')
plt.yticks(fontsize=8, fontname='Arial')
#ax.ticklabel_format(style='plain')  # deactivate scientific notation
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
ax.get_legend().remove()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path_out, 'total_block_relative.pdf'))
pd_fig1.to_csv(os.path.join(path_out, 'total_block_relative.csv'))

print("--finished--")
