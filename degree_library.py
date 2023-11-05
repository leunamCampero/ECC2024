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

def degree_G(G):
    '''
    ----------
     Parameters
     ----------
     G: networkx weighted graph.

     dw: dictionary
         links as the keys and weights as the values
     --------
     Returns:
     --------
     dd: dictionary
         keys as the nodes, and degrees of the respectively nodes as values
    '''
    H = G.degree(weight='weight')
    dd = {}
    for i in list(H):
        dd[i[0]] = i[1]
    return dd

def create_second_graph(G):
    # Create a new graph with the same nodes as G
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes())

    # Add edges to G2 with key=1 and weight=weight/key
    for endpoint1, endpoint2, data in G.edges(data=True):
        G2.add_edge(endpoint1, endpoint2, weight=4)

    return G2


def degree_function(G):
    G2 = create_second_graph(G)
    # Compute the degree of each node in G taking into account the weights of the edges
    degree_G = dict(G.degree(weight='weight'))

    # Compute the degree of each node in G2 taking into account the weights of the edges
    degree_G2 = dict(G2.degree(weight='weight'))

    # Compute the ratio of the degree of each node in G to the degree of the same node in G2
    s = 0
    # s2 = 0
    for node in G.nodes():
        d1 = degree_G[node]
        d2 = degree_G2[node]
        s = d1/d2 + s
    return s/len(G.nodes())