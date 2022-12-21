"""Plots of Nature sustainability paper
"""
import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions
from superblocks.scripts.network import helper_read_write as hp_rw 
from superblocks.scripts.network import helper_plot as hp_plt

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "Arial"

# Remote mode or local
local = False

if local:
    path_folder = "K:/superblocks/03-results_publication/_results_superblockR1"
else:
    path_folder = "K:/superblocks/03-results_publication/_results_superblockR1"


crit_deviations = list(np.array(range(0, 22, 2)) / 10)
crit_deviations = [float(i) for i in crit_deviations]
crit_deviations = [0.0, 0.5, 1.0, 1.5, 2.0]
crit_deviations = [1.0] #Temp

path_result_folder = os.path.join(path_folder, '_plots')
base_name = "classified_edges_withflow"
flows = '_flows'
font_size = 7
hp_rw.create_folder(path_result_folder)

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
    'high_p': '#d95f02'}

# Define NDI flow category values
road_flow_categories = [
    0.05,
    0.15]

# Definition of deviations
plot_deviation_crit = 1.0

# Configs
label_dict = hp_rw.city_labels()  #load labels
factor_distance = 1000
p_factor = 100

cities = [
    #'atlanta',
    #'bankok',
    #'barcelona',
    'berlin',
    #'budapest',
    #'cairo',
    #'hong_kong',
    #'lagos',
    #'london',
    #'madrid',
    #'melbourne',
    #'mexico_city',
    #'paris',
    #'rome',
    #'sydney',
    #'tokyo',
    #'warsaw',
    #'zurich'
    
    'frankfurt',
    'freiburg',
    'hamburg',
    'munchen'
    ]

# =============================================================
# Plots
# =============================================================
sensitivityplot = False
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

    # Final gdf
    df = pd.DataFrame(
        columns=[
            'tot_length', 'l_blocks', 'big_road', 'b_superb', 'b_mini', 'b_miniS', 'b_minicombined', 'pedestrian', 'other',
            'low', 'medium', 'high', 'class', 'type', 'l_city_streets', 'l_superblocks', 'low_superblocks', 'medium_superblocks', 'high_superblocks'],
        index=cities)

    # Load city data
    gdf_dict = {}

    for city in cities:
        print("reading data for city: {} {}".format(city, crit_deviation))
        gdf = gpd.read_file(os.path.join(path_folder, city, flows, "{}_{}.shp".format(base_name, crit_deviation)))

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
# [1] Sensitivity plot road types
#  Plotting across whoe spectrum deviations the share of streets (sensitivity plot)
# https://www.plus2net.com/python/pandas-dataframe-plot-area.php
# ----------------------------------
if sensitivityplot:
    columns_to_plot_roadtypes_small = {
        'b_superb_p': '#7fbf7b', #'#0fc727',
        'b_minicombined_p': '#d1df9a'} #'#ffb525',

    column_names = list(columns_to_plot_roadtypes_small.keys())
    colors = list(columns_to_plot_roadtypes_small.values())

    # ---Plot combined boxplot sensitivity
    fig, ax = plt.subplots()
    path_fig = os.path.join(path_result_folder, "single_sensitivity_boxplots.pdf")

    color_one = colors[0]
    color_two = colors[1]

    # Create boxplot pairs
    pairs = []
    for crit_deviation in crit_deviations:
        df = all_results[crit_deviation][column_names]
        value_superblock = list(df[column_names[0]])
        value_miniblock = list(df[column_names[1]])
        pairs.append([value_superblock, value_miniblock])

    positions = []

    # Means
    gap_between_bars = 0.5
    gap_between_grouped = 1
    position = 0 #* 0 #-1 * gap_between_grouped - gap_between_bars
    df_mean_1 = pd.DataFrame(columns=[column_names[0]])
    df_mean_2 = pd.DataFrame(columns=[column_names[1]])
    for i in range(len(crit_deviations)):
        position += gap_between_bars #1
        bp = ax.boxplot(
            pairs[i], positions=[position, position + gap_between_bars], widths=0.4,
            patch_artist=True,  # fill with color
            autorange=False,
            whis=(0, 100), #formerly range
            zorder=1)
        hp_plt.setBoxColors(bp, color_one, color_two)

        df_mean_1.at[position, column_names[0]] = np.mean(pairs[i][0])
        df_mean_2.at[position + gap_between_bars, column_names[1]] = np.mean(pairs[i][1])

        positions.append(position + gap_between_bars)
        position += gap_between_grouped  #2

    # set axes limits and labels
    ax.axis(ymin=0, ymax=60)
    position_shifted = [i - gap_between_bars/2 for i in positions]

    # Plot mean line
    df_mean_1.plot.line(ax=ax, linewidth=1, style=['.-'], c=color_one, zorder=2)
    df_mean_2.plot.line(ax=ax, linewidth=1, style=['.-'], c=color_two, zorder=2)
    df_mean_1.reset_index().plot(kind='scatter', x='index', y=column_names[0], ax=ax, c='black', zorder=3, s=2)
    df_mean_2.reset_index().plot(kind='scatter', x='index', y=column_names[1], ax=ax, c='black', zorder=3, s=2)

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='major', linewidth=0.6, color='lightgrey', linestyle='--')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks(position_shifted, minor=False)
    ax.set_xticklabels([str(j) for j in crit_deviations], fontname='Arial', fontdict=None, minor=False, fontsize=font_size)
    #ax.set_yticklabels(fontsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    ax.tick_params(axis='x', direction='inout', length=6, size=font_size)
    plt.xlabel("Geometric deviation factor (f)", labelpad=5, weight='bold', size=font_size)
    plt.ylabel("Street length (%)", labelpad=5, weight='bold', size=font_size)
    #df_mean_1.to_csv("{}.csv".format(path_fig[:-4]))
    ax.get_legend().remove()
    fig.set_size_inches(hp_rw.cm2inch(9, 9))
    plt.tight_layout()

    plt.savefig(path_fig, bbox_inches="tight")
    '''prnt(":")

    # ---Plot combined sensitivity plot
    fig, ax = plt.subplots()
    for city in cities: 
        city = label_dict[city]
        df_sensitivity = pd.DataFrame(index=crit_deviations, columns=columns_to_plot_roadtypes_small.keys())
        for crit_deviation in crit_deviations:
            df = all_results[crit_deviation][columns_to_plot_roadtypes_small.keys()]
            df_sensitivity.loc[crit_deviation] = df.loc[city].values.tolist()

        df_sensitivity.index = df_sensitivity.index #* p_factor
        df_sensitivity.plot.line(ax=ax, color=columns_to_plot_roadtypes_small.values(), linewidth=1, style=['--', '-'])
    plt.ylim(0, 60)
    plt.xlim(0, 2)
    plt.xlabel("Geometric deviation factor (f)", labelpad=5, weight='bold', size=font_size)
    plt.ylabel("Street length (%)", labelpad=5, weight='bold', size=font_size)
    print(df_sensitivity)

    path_fig = os.path.join(path_result_folder, "single_sensitivity.pdf")
    hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
    ax.get_legend().remove()
    fig.set_size_inches(hp_rw.cm2inch(9, 9))
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight")


    # --Individual plots
    for colum_to_plot, color_colum in columns_to_plot_roadtypes_small.items():
        fig, ax = plt.subplots()
        for city in cities:
            city = label_dict[city]
            df_sensitivity = pd.DataFrame(index=crit_deviations, columns=[colum_to_plot])
            for crit_deviation in crit_deviations:
                df = all_results[crit_deviation][colum_to_plot]
                df_sensitivity.loc[crit_deviation] = df.loc[city]

            df_sensitivity.index = df_sensitivity.index * p_factor
            df_sensitivity.plot.line(ax=ax, color=color_colum, linewidth=1, style=['--'])
        plt.ylim(0, 60)
        plt.xlim(0, 200)
        plt.xlabel("Geometric deviation factor (f)", labelpad=5, weight='bold', size=font_size)
        plt.ylabel("Street length (%)", labelpad=5, weight='bold', size=font_size)

        path_fig = os.path.join(path_result_folder, "single_sensitivity_{}.pdf".format(colum_to_plot))
        hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
        ax.get_legend().remove()
        fig.set_size_inches(hp_rw.cm2inch(9, 9))
        plt.tight_layout()
        #plt.show()
        plt.savefig(path_fig, bbox_inches="tight")

    # ----Plot individual sensitivity plots
    for city in cities: 
        city = label_dict[city]
        fig, ax = plt.subplots()
        df_sensitivity = pd.DataFrame(index=crit_deviations, columns=columns_to_plot_roadtypes.keys())
        for crit_deviation in crit_deviations:
            df = all_results[crit_deviation][columns_to_plot_roadtypes.keys()]
            df_sensitivity.loc[crit_deviation] = df.loc[city].values.tolist()

        df_sensitivity.index = df_sensitivity.index * p_factor
        df_sensitivity.plot.area(stacked=True, ax=ax, color=columns_to_plot_roadtypes.values(), linewidth=0)
        plt.ylim(0, 100)
        plt.xlim(0, 200)
        #plt.title("Sensitivity claculation: {}".format(city))
        plt.xlabel("Geometric deviation factor (f)", labelpad=5, weight='bold', size=font_size)
        plt.ylabel("Street length (%)", labelpad=5, weight='bold', size=font_size)
        print("---- {} ----".format(city))
        print(df_sensitivity)

        path_fig = os.path.join(path_result_folder, "sensitivity_{}.pdf".format(city))
        hp_rw.export_legend(ax.legend(), "{}__legend.pdf".format(path_fig[:-4]))
        ax.get_legend().remove()

        fig.set_size_inches(hp_rw.cm2inch(7, 5))
        plt.tight_layout()
        plt.savefig(path_fig, bbox_inches="tight")
        df_sensitivity.to_csv("{}.csv".format(path_fig[:-4]))
    '''

# ----------------------------------
# [2] - Plotting flow category percentage of roads of superblocks (DFI)
# ----------------------------------
if flow_percentages:

    # ----superblocks only
    fig, ax = plt.subplots()
    df = all_results[plot_deviation_crit]
    df = df.sort_values(['low_p_superblocks', 'medium_p_superblocks'], ascending=False)     # Sorting
    df[['low_p_superblocks', 'medium_p_superblocks', 'high_p_superblocks']].plot(
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), fontsize=font_size, zorder=1)

    plt.xlim(0, 100)
    plt.xlabel("Superblock and miniblock street (%)", labelpad=5, fontname='Arial', weight='bold', size=font_size)

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
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), fontsize=font_size, zorder=1)
    plt.xlim(0, 100)
    plt.xlabel("Superblock and miniblock street (%)", labelpad=5, fontname='Arial', weight='bold', size=font_size)

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
    df[['low', 'medium', 'high']].plot(ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), fontsize=font_size, zorder=1)

    plt.xlabel("Superblock and miniblock street (km)", labelpad=5, fontname='Arial', weight='bold', size=font_size)

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
        ax=ax, kind='barh', stacked=True, linewidth=0.5, rot=0, edgecolor='black', color=flow_cat.values(), fontsize=font_size, zorder=1)

    plt.xlabel("Street length of superblocks (km)", labelpad=5, weight='bold', size=font_size)

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
# [2] Plotting percetange of road flow types https://towardsdatascience.com/stacked-bar-charts-with-pythons-matplotlib-f4020e4eb4a7
# ---------------------------------------------------------------------------------------------------
'''fig, ax = plt.subplots(figsize=font_size_rw.cm2inch(10, 9))
columns_to_plot = ['low', 'medium', 'high']

# Calculate error bars
df_upper = all_results[upper]
df_lower = all_results[lower]
for column_to_plot in columns_to_plot:
    df['yerr_min'] = df_lower[column_to_plot]
    df['yerr_max'] = df_upper[column_to_plot]

df[columns_to_plot].plot(kind='bar', ax=ax, stacked=True, linewidth=0.5, edgecolor='black', color=['#fdae61', '#ffffbf', '#abdda4']) #,
#yerr=df[['yerr_min', 'yerr_max']].T.values)

plt.xlim(0, 100)
plt.ylabel("road flow index (% of total road)", labelpad=20, weight='bold', size=font_size
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
'''
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
        kind='barh', ax=ax, stacked=True, linewidth=0.5, edgecolor='black', fontsize=font_size,
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
    plt.xlabel("Street length (%)", labelpad=5, weight='bold', size=font_size)
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
        kind='barh', ax=ax, stacked=True, linewidth=0.5, edgecolor='black', fontsize=font_size,
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
    plt.xlabel("Street length (km)", labelpad=5, weight='bold', size=font_size)
    plt.tight_layout()
    df.to_csv("{}.csv".format(path_fig[:-4]))
    plt.savefig(path_fig, bbox_inches="tight")



print("___finished____")