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
import networkx as nx
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

def create_second_graph(G):
    # Create a new graph with the same nodes as G
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())

    # Add edges to G2 with key=1 and weight=weight/key
    for endpoint1, endpoint2, data in G.edges(data=True):
        G2.add_edge(endpoint1, endpoint2, weight=4)

    return G2

def diameter_function(G, parameter):
    G2 = create_second_graph(G)
    # Create a new graph H with the largest connected component of the subgraph G where all the edges have weight greater or equal to 3
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d['weight'] >= parameter:
            H.add_edge(u, v, weight=d['weight'])

    # Get the largest connected component of H
    largest_cc = max(nx.connected_components(H), key=len)
    W = H.subgraph(largest_cc)
    # print(len(largest_cc))
    # Get the largest connected component of G2
    largest_cc_G2 = max(nx.connected_components(G2), key=len)
    # print(len(largest_cc_G2))
    W1 = G2.subgraph(largest_cc_G2)
    # Create a color map for the edges
    # ratio = []
    # W_edges = W.edges()
    # for u, v in G.edges():
    #     if ((u,v) in W_edges or (v, u) in W_edges):
    #         ratio.append(100)
    #     else:
    #         ratio.append(0)
    return (len(W.nodes()) + sum(d['weight'] for u, v, d in W.edges(data=True)))/(len(W1.nodes()) + sum(d['weight'] for u, v, d in W1.edges(data=True)))
    # return (len(W.nodes()) )/(len(W1.nodes()) )