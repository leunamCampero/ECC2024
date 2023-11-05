from pathlib import Path
from ipyleaflet import Map, LayersControl, LayerGroup, Polygon, GeoJSON, LegendControl, FullScreenControl, basemaps, basemap_to_tiles
from ipywidgets.embed import embed_minimal_html
import json
import random
from tqdm.auto import tqdm
import pandas as pd
from itertools import chain
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
#from shapely.geometry import Point, LineString
from geojson import Feature, LineString
import numpy as np
import geopandas as gpd
from geojson_length import calculate_distance, Unit
from shapely.geometry import Point as geoPoint, Polygon as geoPolygon
from shapely import wkt
import geojson
from pyproj import Geod
from area import area
import random
import networkx as nx
import matplotlib.pyplot as plt
import math

def create_second_graph(G):
    # Create a new graph with the same nodes as G
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())

    # Add edges to G2 with key=1 and weight=weight/key
    for endpoint1, endpoint2, data in G.edges(data=True):
        G2.add_edge(endpoint1, endpoint2, weight=1)

    return G2


def modify_graph(G):
    # Define a dictionary to map the old values to the new values
    value_map = {1: 4, 2: 3, 3: 2, 4: 1}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    # Iterate through each edge in the graph and modify the values
    for endpoint1, endpoint2, data in G.edges(data=True):
        # Modify the value of the weight
        new_weight = value_map[data['weight']]
        H.add_edge(endpoint1, endpoint2, weight=new_weight)

    return H


def closeness_function(G):
    G2 = create_second_graph(G)
    H = modify_graph(G)
    
    closeness_G = nx.closeness_centrality(H, u=None, distance='weight', wf_improved=True)

    # Compute the closeness centrality of each node in G2 taking into account the weights of the edges
    closeness_G2 = nx.closeness_centrality(G2, u=None, distance='weight', wf_improved=True)

    # Compute the ratio of the closeness centrality of each node in G to the closeness centrality of the same node in G2
    # ratio = []
    # ratio = [(closeness_G[node]/closeness_G2[node])*100 for node in G.nodes()]
    s = 0
    # s2 = 0
    for node in H.nodes():
        c1 = closeness_G[node]*100
        c2 = closeness_G2[node]*100
        # s1 += c1
        s = c1/c2 + s
        # ratio.append(c1/c2)
    return s/len(H.nodes()) 