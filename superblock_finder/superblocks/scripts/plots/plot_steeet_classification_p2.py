"""Plots of greening publication
"""
import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_read_write as hp_rw 
from superblocks.scripts.network import helper_plot as hp_plt

# Local
path_folder = "C:/_results_swiss_communities/_results"
path_result_folder = "H:/09-papers/2021-superblocks_urban_greening/_results_simulation"
path_communities = "C:/DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"

# Server\5192\blocks\0.5
#path_folder = "J:/Sven/greening/_results_swiss_communities/_results"
#path_result_folder = "H:/09-papers/2021-superblocks_urban_greening/_results_simulation"
#path_communities = "J:/Sven/_DATA/are_gemeindetypologie/data/ARE_GemTyp00_9.shp"

crit_deviations = [0.5]

base_name = "classified_edges_withflow"
flows = '_flows'

hp_rw.create_folder(path_result_folder)

gemeinden = gpd.read_file(path_communities)
gemeinden = gemeinden.loc[gemeinden['NAME'].isin(['Grosszentren'])]  #, 'Mittelzentren'])]
#segmentation_IDs = list(gemeinden['BFS_NO'])
#segmentation_IDs.sort()

#columns_to_plot_roadtypes = ['big_road', 'b_superb', 'b_mini', 'b_miniS']
columns_to_plot_roadtypes = {
    'b_superb_p': '#7fbf7b', #'#0fc727',
    'b_minicombined_p': '#d1df9a', #'#ffb525',
    'pedestrian_p': '#e9a3c9',
    'big_road_p': '#ef8a62', #'#f13f3f',
    'other_p': '#A9A9A9', # '#9b9b9b',
    }
columns_to_plot_absolutes = ['b_superb', 'b_minicombined', 'pedestrian', 'big_road', 'other']

flow_cat = {
    'low_p': '#66c2a5' ,
    'medium_p': '#8da0cb',
    'high_p': '#d95f02'} #'#fc8d62'}

# Define NDI flow category values
road_flow_categories = [
    0.05,
    0.15]

# Definition of deviations
plot_deviation_crit = 0.5

# Configs
label_dict = hp_rw.city_labels_ch()
factor_distance = 1000
p_factor = 100

#cities = [6621, 1061] #[261, 351, 1061, 2701]
cities = [230, 261, 351, 1061, 2701, 3203, 5192, 5586, 6621]

# =============================================================
# Plots
# =============================================================
sensitivityplot = True
flow_percentages = True
flow_absolute = True
plot_road_percentage = True
plot_road_absolute = True

# =============================================================
# LAOD DATA
# =============================================================
# All results container
all_results = {}

for crit_deviation in crit_deviations:

    cities_label = [list(gemeinden.loc[gemeinden['BFS_NO'] == i]['GDE_NAME'])[0] for i in cities]
    df = pd.DataFrame(
        columns=[
            'tot_length', 'l_blocks', 'big_road', 'b_superb', 'b_mini', 'b_miniS', 'b_minicombined', 'pedestrian', 'other',
            'low', 'medium', 'high', 'class', 'type', 'l_city_streets', 'l_superblocks', 'low_superblocks', 'medium_superblocks', 'high_superblocks'],
        index=cities_label)

    gdf_dict = {}

    for city in cities:
        print("reading data for city: {} {}".format(city, crit_deviation))
        gdf = gpd.read_file(os.path.join(path_folder, str(city), flows, "{}_{}.shp".format(base_name, crit_deviation)))

        # Get label
        gemeinden_label = list(gemeinden.loc[gemeinden['BFS_NO'] == city]['GDE_NAME'])[0]
        city = gemeinden_label

        # Classify flow of resilience value
        gdf = flow_algorithm_functions.classify_flow_cat(
            gdf,
            label_out='flow_cat',
            label='flow_ov',
            input_type='gdf',
            cats=road_flow_categories)

        # Remove service streets
        gdf_selection = gdf.loc[gdf['tags.highway'] != 'service']

        # Add calculations to result container
        df['big_road'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'big_road'].geometry.length) / factor_distance
        df['b_superb'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'b_superb'].geometry.length) / factor_distance
        df['b_mini'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'b_mini'].geometry.length) / factor_distance
        df['b_miniS'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'b_miniS'].geometry.length) / factor_distance
        df['b_minicombined'][city] = np.sum(gdf_selection.loc[(gdf_selection['final'] == 'b_miniS') | (gdf_selection['final'] == 'b_mini')].geometry.length) / factor_distance
        df['pedestrian'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'pedestrian'].geometry.length) / factor_distance
        df['other'][city] = np.sum(gdf_selection.loc[gdf_selection['final'] == 'other'].geometry.length) / factor_distance
        df['l_city_streets'][city] = np.sum(gdf_selection.geometry.length) / factor_distance
        df['tot_length'][city] = np.sum(gdf.geometry.length.tolist()) / factor_distance

        # Select only superblock&miniblocks
        gdf_superblocks = gdf_selection.loc[gdf_selection['final'].isin(['b_superb'])]
        df['l_superblocks'][city] = np.sum(gdf_superblocks.geometry.length.tolist()) / factor_distance
        gdf_blocks = gdf_selection.loc[gdf_selection['final'].isin(['b_superb', 'b_mini', 'b_miniS'])]
        df['l_blocks'][city] = np.sum(gdf_blocks.geometry.length.tolist()) / factor_distance

        # FLOW of superblocks and miniblocks
        df['low'][city] = np.sum(gdf_blocks.loc[gdf_blocks['flow_cat'] == 'low'].geometry.length) / factor_distance
        df['medium'][city] = np.sum(gdf_blocks.loc[gdf_blocks['flow_cat'] == 'medium'].geometry.length) / factor_distance
        df['high'][city] = np.sum(gdf_blocks.loc[gdf_blocks['flow_cat'] == 'high'].geometry.length) / factor_distance

        # FLOW of superblocks and miniblocks
        df['low_superblocks'][city] = np.sum(gdf_superblocks.loc[gdf_superblocks['flow_cat'] == 'low'].geometry.length) / factor_distance
        df['medium_superblocks'][city] = np.sum(gdf_superblocks.loc[gdf_superblocks['flow_cat'] == 'medium'].geometry.length) / factor_distance
        df['high_superblocks'][city] = np.sum(gdf_superblocks.loc[gdf_superblocks['flow_cat'] == 'high'].geometry.length) / factor_distance

    # Calculations
    df['big_road_p'] = df['big_road'] / df['l_city_streets'] * p_factor
    df['b_superb_p'] = df['b_superb'] / df['l_city_streets'] * p_factor
    df['b_mini_p'] = df['b_mini'] / df['l_city_streets'] * p_factor
    df['b_miniS_p'] = df['b_miniS'] / df['l_city_streets'] * p_factor
    df['b_minicombined_p'] = df['b_minicombined'] / df['l_city_streets'] * p_factor
    df['pedestrian_p'] = df['pedestrian'] / df['l_city_streets']  * p_factor
    df['other_p'] = df['other'] / df['l_city_streets'] * p_factor

    # Calculate percentage of street length of flow category of blocks
    df['low_p'] = df['low'] / df['l_blocks'] * p_factor
    df['medium_p'] = df['medium'] / df['l_blocks'] * p_factor
    df['high_p'] = df['high'] / df['l_blocks'] * p_factor

    df['low_p_superblocks'] = df['low_superblocks'] / df['l_superblocks'] * p_factor
    df['medium_p_superblocks'] = df['medium_superblocks'] / df['l_superblocks'] * p_factor
    df['high_p_superblocks'] = df['high_superblocks'] / df['l_superblocks'] * p_factor

    # Renaming for plotting
    df = df.rename(index=label_dict)  #Rename Index
    all_results[crit_deviation] = df

# ----------------------------------
# [2] - Plotting flow category percentage of roads of superblocks (DFI)
# ----------------------------------
if flow_percentages:

    # ----superblocks only
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['low_p_superblocks', 'medium_p_superblocks'], ascending=False)     # Sorting
    df[['low_p_superblocks', 'medium_p_superblocks', 'high_p_superblocks']].plot(
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), zorder=1)

    plt.xlim(0, 100)
    plt.xlabel("Superblock and miniblock street (%)", labelpad=5, weight='bold', size=10)

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    path_fig = os.path.join(path_result_folder, "flow_p_superblock.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()
    fig.set_size_inches(hp_rw.cm2inch(9.5, 9))
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))

    # ----superblocks and miniblocks
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['low_p', 'medium_p'], ascending=False)     # Sorting
    df[['low_p', 'medium_p', 'high_p']].plot(
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), zorder=1)
    plt.xlim(0, 100)
    plt.xlabel("Superblock and miniblock street (%)", labelpad=5, weight='bold', size=10)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    path_fig = os.path.join(path_result_folder, "flow_p_superblock_and_miniblock.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()

    fig.set_size_inches(hp_rw.cm2inch(9.5, 9))
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))

# ----------------------------------
# [2b] Absolute - Plotting flow category of roads of superblocks (DFI)
# ----------------------------------
if flow_absolute:
    # ---Superblocks and miniblocks
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['low', 'medium'], ascending=False)     # Sorting
    df[['low', 'medium', 'high']].plot(ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), zorder=1)

    plt.xlabel("Superblock and miniblock street (km)", labelpad=5, weight='bold', size=10)

    path_fig = os.path.join(path_result_folder, "flow_absolutes_superblock_and_miniblock.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.set_size_inches(hp_rw.cm2inch(9.5, 9))
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))


    # ---Superblocks only
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['low_superblocks', 'medium_superblocks'], ascending=False)     # Sorting
    df[['low_superblocks', 'medium_superblocks', 'high_superblocks']].plot(
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), zorder=1)

    plt.xlabel("Street length of superblocks (km)", labelpad=5, weight='bold', size=10)

    path_fig = os.path.join(path_result_folder, "flow_absolutes_superblocks.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.set_size_inches(hp_rw.cm2inch(9.5, 9))
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))


# ---------------------------------------------------------------------------------------------------
# [3] Plotting roads type
#https://stackoverflow.com/questions/64796315/how-to-plot-min-max-bars-with-a-bar-plot
# ---------------------------------------------------------------------------------------------------
if plot_road_percentage:

    # ---Super and miniblocks
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['b_superb_p', 'b_minicombined_p', 'pedestrian_p'], ascending=False)
    df[columns_to_plot_roadtypes.keys()].plot(
        kind='barh', ax=ax, stacked=True, linewidth=0.5, edgecolor='black',
        color=columns_to_plot_roadtypes.values())

    path_fig = os.path.join(path_result_folder, "road_types_relative.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.set_size_inches(hp_rw.cm2inch(10, 9))
    plt.xlim(0, 100)
    plt.xlabel("Street length (%)", labelpad=5, weight='bold', size=10)
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))

# ---------------------------------------------------------------------------------------------------
# [4] absolute length of road types
#https://stackoverflow.com/questions/64796315/how-to-plot-min-max-bars-with-a-bar-plot
# ---------------------------------------------------------------------------------------------------
if plot_road_absolute:
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['b_superb', 'b_minicombined', 'pedestrian'], ascending=False)
    df[columns_to_plot_absolutes].plot(
        kind='barh', ax=ax, stacked=True, linewidth=0.5, edgecolor='black',
        color=columns_to_plot_roadtypes.values())

    path_fig = os.path.join(path_result_folder, "road_types_absolute.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linewidth=0.7, color='grey', linestyle='-')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.set_size_inches(hp_rw.cm2inch(10, 9))
    plt.xlabel("Street length (km)", labelpad=5, weight='bold', size=10)
    plt.tight_layout()

    plt.savefig(path_fig, bbox_inches="tight")
    df.to_csv("{}.csv".format(path_fig[:-4]))

print("___finished____")