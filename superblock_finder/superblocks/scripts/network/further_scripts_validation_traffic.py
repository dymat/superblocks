"""Load actual traffic data and provide validatino plot
"""
import os
import sys
path_superblocks = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(path_superblocks)

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy import stats

from superblocks.scripts.network import helper_network as hp_net
from superblocks.scripts.network import helper_read_write as hp_rw
from superblocks.scripts.network import helper_osm as hp_osm
from superblocks.scripts.network import flow_algorithm_functions as flow_algorithm_functions

to_crs_meter = 32632

# Paths
path_out = "C:/_results_superblock/zurich/_flows"
path_traffic_flow_2056 = "K:/superblocks/01-data/traffic_data/traffic_flow_raw.shp"
path_traffic_flow_32632 = "K:/superblocks/01-data/traffic_data/traffic_flow_32632.shp"

# Path to base flow
path_own_network = "C:/_results_superblock/zurich/_flows/base_flow_edmunds_recalculated.shp"

project_streets = False
assign_flow_attributes = False
plot_figure = True

if project_streets:
    street_raw = gpd.read_file(path_traffic_flow_2056)
    street_raw = hp_net.project(street_raw, to_crs_meter)
    street_raw.to_file(path_traffic_flow_32632)

if assign_flow_attributes:

    # Assign traffic flow to own network
    gdf_street = gpd.read_file(path_own_network)
    G_street = hp_rw.gdf_to_nx(gdf_street)

    bb = hp_osm.BB(
        ymax=max(gdf_street.geometry.bounds.maxy),
        ymin=min(gdf_street.geometry.bounds.miny),
        xmax=max(gdf_street.geometry.bounds.maxx),
        xmin=min(gdf_street.geometry.bounds.minx))
    bb_gdf = bb.as_gdf(crs_orig=to_crs_meter)

    # Load flow graph
    traffic_data = gpd.read_file(path_traffic_flow_32632)
    traffic_data = hp_net.clip_outer_polygons(traffic_data, bb_gdf.geometry[0])

    G_base = hp_rw.gdf_to_nx(traffic_data)

    # Clip network
    labels = ['DWV_FZG', 'DWV_PW']
    G_assigned = hp_net.assign_attribute_by_largest_intersection(
        G_street,
        G_base,
        min_intersection_d=10,
        crit_buffer=4,
        labels=labels)
    _, edges = hp_rw.nx_to_gdf(G_assigned)

    # Normalize labels
    for label in labels:
        max_value = max(edges[label])
        print("Maximum traffic flow value: {}".format(max_value))
        edges['{}_norm'.format(label)] = round(edges[label] / max_value, 4)

    edges['l'] = edges.geometry.length
    edges.to_file(os.path.join(path_out, "G_with_traffic_raw.shp"))


plot_average_capacity_per_type = False
if plot_average_capacity_per_type:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.boxplot.html
    # Read data
    gdf_street = gpd.read_file(os.path.join(path_out, "G_with_traffic_raw.shp"))
    grouped = gdf_street[['DWV_FZG', 'tags.highway']].groupby('tags.highway')
    grouped.boxplot(subplots=False, rot=5, fontsize=4)
    plt.show()

    # Actual traffic per lane
    #gdf_street2 = gdf_street.set_index('DWV_FZG', drop=True)
    #grouped = gdf_street2[['capacity', 'DWV_FZG']].groupby('capacity')
    #grouped = gdf_street[['DWV_FZG', 'capacity']].groupby('capacity')
    #grouped = grouped.set_index('DWV_FZG')
    #grouped.boxplot(subplots=False, rot=5, fontsize=7)
    #plt.show()

plot_figure_stacked_different_types = True
if plot_figure_stacked_different_types:


    # -----------------------
    # Group traffic data into three classes 
    # as well as group flow_ov into three classes --> ch
    # -----------------------
    label = 'DWV_FZG'
    gdf_street = gpd.read_file(os.path.join(path_out, "G_with_traffic_raw.shp"))


    min_length = 20  # [m]
    gdf_street = gdf_street.loc[gdf_street.geometry.length > min_length]
    #gdf_street = gdf_street.loc[~gdf_street['DWV_FZG_no'].isna()]  #TODO: improve
    
    # CREATE STREET CLASSSES (LOW, MEDIUM, HIGH)
    gdf_street = flow_algorithm_functions.classify_flow_cat(
        gdf_street, label_out='cl_DWV', label='DWV_FZG_no', input_type='gdf', cats=[0.15, 0.3])

    gdf_street = flow_algorithm_functions.classify_flow_cat(
        gdf_street, label_out='cl_flov', label='flow_ov', input_type='gdf', cats=[0.15, 0.3])

    '''gdf_street['cl_DWV'] = 0
    gdf_street.at[gdf_street['flow_ov'] <= 0.2, 'cl_DWV'] = 0
    gdf_street.at[(gdf_street['flow_ov'] > 0.2) & (gdf_street['flow_ov'] <= 0.4), 'cl_DWV'] = 1
    gdf_street.at[(gdf_street['flow_ov'] > 0.4), 'cl_DWV'] = 2
    print("Number class 0: {}".format(gdf_street.loc[gdf_street['cl_DWV'] == 0].shape[0]))
    print("Number class 1: {}".format(gdf_street.loc[gdf_street['cl_DWV'] == 1].shape[0]))
    print("Number class 2: {}".format(gdf_street.loc[gdf_street['cl_DWV'] == 2].shape[0]))

    gdf_street['cl_flov'] = 0
    gdf_street.at[gdf_street['DWV_FZG_norm'] <= 0.2, 'cl_flov'] = 0
    gdf_street.at[(gdf_street['DWV_FZG_norm'] > 0.2) & (gdf_street['DWV_FZG_norm'] <= 0.4), 'cl_flov'] = 1
    gdf_street.at[(gdf_street['DWV_FZG_norm'] > 0.5), 'cl_flov'] = 2

    print("Number class 0: {}".format(gdf_street.loc[gdf_street['cl_flov'] == 0].shape[0]))
    print("Number class 1: {}".format(gdf_street.loc[gdf_street['cl_flov'] == 1].shape[0]))
    print("Number class 2: {}".format(gdf_street.loc[gdf_street['cl_flov'] == 2].shape[0]))'''

    # Sum the length and have classes in own columns
    gdf_streetsum = gdf_street.groupby(['cl_flov', 'cl_DWV'], as_index=False)['l'].sum()
    #gdf_streetsum = gdf_streetsum.set_index('cl_DWV', drop=True)
    gdf_streetsum2 = gdf_streetsum.set_index('cl_DWV', drop=True)
    print("-----")
    print(gdf_streetsum)
    #gdf_streetsum.groupby(['class_flowov', 'class_traffic']).unstack().plot(kind='bar',stacked=True)
    
    sub_df = gdf_street.groupby(['cl_DWV', 'cl_flov'])['l'].sum().unstack()
    sub_df = sub_df[['low', 'middle', 'high']]  # Sort columns
    sub_df = sub_df.loc[['low', 'middle', 'high']]  # Sort index    
    sub_df.plot(kind='bar', stacked=True)

    print("-----")
    print(sub_df)
    plt.show()
    #gdf_streetsum = gdf_streetsum.sum(axis=1).sum().sum()
    #df.groupby(['Fruit','Name'])['Number'].sum()
    #gdf_streetsum2.plot(x='class_flowov', kind='bar', stacked=True)
    #gdf_street.pl https://stackoverflow.com/questions/23415500/pandas-plotting-a-stacked-bar-chart
    #gdf_street_grouped = gdf_street[['class_flowov', 'class_traffic', 'l']].groupby(['class_traffic']) #['class_traffic'].count().unstack('class_traffic').fillna(0)
    #gdf_street_grouped[['class_flowov','class_traffic']].plot(kind='bar', stacked=True)
    #df2 = df.groupby(['Name', 'Abuse/NFF'])['Name'].count().unstack('Abuse/NFF').fillna(0)
    #df2[['abuse','nff']].plot(kind='bar', stacked=True)



plot_scatter_flow_ov_centrality = True
if plot_scatter_flow_ov_centrality:

    # Plot centrality of each edge versus flwo of eac heacthed
    # ==============================================================
    gdf_street = gpd.read_file(os.path.join(path_out, "G_with_traffic_raw.shp"))
    df_street_data = pd.DataFrame(gdf_street)

    # TODO: TODO:
    road_flow_categories = [
        0.05,
        0.15]
    # CREATE STREET CLASSSES (LOW, MEDIUM, HIGH)
    gdf_street = flow_algorithm_functions.classify_flow_cat(
        gdf_street, label_out='cl_DWV', label='DWV_FZG_no', input_type='gdf', cats=[0.15, 0.3])

    gdf_street = flow_algorithm_functions.classify_flow_cat(
        gdf_street, label_out='cl_flov', label='flow_ov', input_type='gdf', cats=[0.15, 0.3])
    gdf_street.to_file("C:/_scrap/B.shp")
    # Only select streets which also have traffic data
    df_street_data = df_street_data.loc[~df_street_data['DWV_FZG_no'].isna()]  #TODO: improve

    ##df_street_data = df_street_data.loc[df_street_data['flow_ov']>0]  #TEST REMOVE

    container = [
        ('flow_ov', 'centrality'),
        ('flow_ov', 'DWV_FZG_no')
    ]

    for label1, label2 in container:
        fig, ax = plt.subplots(figsize=hp_rw.cm2inch(9, 9))
        df_street_data.plot(x=label1, y=label2, kind='scatter', ax=ax)

        # Linear regression
        linear_regressor = LinearRegression()
        x = df_street_data[label1].values
        y = df_street_data[label2].values
        linear_regressor.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        Y_pred = linear_regressor.predict(x.reshape(-1, 1))
        plt.plot(x, Y_pred, color='red')
        plt.xlabel(label1)
        plt.ylabel(label2)
        res = stats.linregress(x, y)
        print("R-squared: {}".format(round(res.rvalue**2, 5)))
        plt.tight_layout()
        plt.show()

    # -----------------------
    # Plot with raw traffic cata
    # -----------------------
    # Read data
    gdf_street = gpd.read_file(os.path.join(path_out, "G_with_traffic_raw.shp"))

    fig, ax = plt.subplots(figsize=hp_rw.cm2inch(9, 9))

    label = 'DWV_FZG'
    label2 = 'flow_ov'

    norm_label = 'norm_{}'.format(label)

    # Get only roads which have traffic from traffic model
    gdf_street_data = gdf_street.loc[gdf_street[label] > 0]

    # Minimum length
    min_length = 20  # [m]
    gdf_street_data = gdf_street_data.loc[gdf_street_data.geometry.length > min_length]

    # Normalize traffic flow
    max_value = max(gdf_street[label])
    print("Maximum traffic flow value: {}".format(max_value))
    gdf_street_data[norm_label] = round(gdf_street[label] / max_value, 4)

    gdf_street_data['diff'] = gdf_street_data[norm_label] - gdf_street_data[label2]
    gdf_street_data.to_file("C:/_scrap/A.shp")

    # Convert to dataframe
    df_street_data = pd.DataFrame(gdf_street_data)
    df_street_data = df_street_data.reset_index(drop=True)
    #df_street_data.plot(y=norm_label, color='red', kind='line', ax=ax)
    #df_street_data.plot(y=label2, color='blue', kind='line', ax=ax)
    df_street_data.plot(x=label2, y=norm_label, kind='scatter', ax=ax)

    # Linear regression
    linear_regressor = LinearRegression()
    x = df_street_data[label2].values
    y = df_street_data[norm_label].values
    linear_regressor.fit(x.reshape(-1,1), y.reshape(-1,1))
    Y_pred = linear_regressor.predict(x.reshape(-1,1))
    plt.plot(x, Y_pred, color='red')

    # Calculate R-Value
    res = stats.linregress(x, y)
    print("R-squared: {}".format(round(res.rvalue**2, 5)))


    plt.show()

print("____finish___")
    