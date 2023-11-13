import geopandas as gpd
import matplotlib.pyplot as plt
import momepy
import networkx as nx
from contextily import add_basemap
from libpysal import weights
from shapely.geometry import LineString
import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
import os
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy import linalg
from closeness_library import closeness_function
from degree_library import degree_function
from diameter_library import diameter_function

bikes = gpd.read_file(r'C:\\Users\\camperom\\safety-of-bike-networks\\Code\\OSM_extract\\GovermentData\\velo_mobilites_m.geojson')
bikes = momepy.extend_lines(bikes, 0.00001)
bikes.geometry = momepy.close_gaps(bikes, 0.00001)
df1 = pd.DataFrame(bikes)
list(bikes['geometry'][0].coords)
def coords(geom):
    return list(geom.coords)
coords = bikes.apply(lambda row: coords(row.geometry), axis=1)
coordn = coords.to_numpy()
pandadata = pd.DataFrame(bikes)
numpydata = pandadata.to_numpy()
final_points = []
G = nx.Graph()
c = 0
dw = {}
for i in coordn:
    if i != []:
        final_points.append(i[0])
        final_points.append(i[len(i)-1])
        for j in range(len(i)-1):
            if (numpydata[c][4] == 'chronovelo'):
                G.add_edge(i[j],i[j+1], weight = 4)
                dw[(i[j],i[j+1])] = 4
                dw[(i[j + 1], i[j])] = 4
            if (numpydata[c][4] == 'veloseparatif'):
                G.add_edge(i[j],i[j+1], weight = 3)
                dw[(i[j], i[j + 1])] = 3
                dw[(i[j + 1], i[j])] = 3
            if (numpydata[c][4] == 'veloconseille'):
                G.add_edge(i[j],i[j+1], weight = 2)
                dw[(i[j], i[j + 1])] = 2
                dw[(i[j + 1], i[j])] = 2
            if (numpydata[c][4] == 'velodifficile'):
                G.add_edge(i[j],i[j+1], weight = 1)
                dw[(i[j], i[j + 1])] = 1
                dw[(i[j + 1], i[j])] = 1
        c = c + 1
# dw = duplicate_d(dw)
for i in list(G.nodes):
    nh = [n for n in G.neighbors(i)]
    if ((len(nh) == 2) and (i not in final_points)):
        G.add_edge(nh[0], nh[1], weight = dw[(i,nh[0])])
        G.remove_node(i)
for e in list(G.edges):
    if (e[0] == e[1]):
        G.remove_edge(*e)
pos = {n: [n[0], n[1]] for n in list(G.nodes)}

GG = G.copy()
#-----------------------------------------------------------------PARAMETER IS THE SAFETY LEVEL
parameter = 1

# Compute the betweenness centrality of each edge
edge_betweenness = nx.edge_betweenness_centrality(GG, weight='weight')

# Filter out the edges that don't have a weight of 2
edges_with_weight_p = [(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter]

# Sort the remaining edges by their betweenness centrality
edges_sorted_by_betweenness = sorted(edges_with_weight_p, key=lambda edge: edge_betweenness[edge], reverse=True)

num_sets = 20
M = len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter])

base_size = M//num_sets  # number of elements in each set
extra = M % num_sets  # number of sets that should have one extra element

NV = [base_size + 1 if i < extra else base_size for i in range(num_sets)]


#-------------------------------------------------------------ALGORITHM 1
GG = G.copy()

# Create the subgraph H
# H = GG.edge_subgraph(edges_with_weight_2)
edges_sorted_by_betweenness_a = edges_sorted_by_betweenness.copy()
border_sets = [] 
# Perform 20 iterations
for i in range(20):
    # Compute the largest connected component of H
    weights_p = [(u, v) for u, v, data in GG.edges(data=True) if data['weight'] >= parameter + 1]

    H = GG.edge_subgraph(weights_p)
    largest_cc = max(nx.connected_components(H), key=len)

    # Find the edges of weight 1 that are connected to the largest connected component
    edges_with_weight_1 = [(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter and (u in largest_cc or v in largest_cc)]

    # Sort the edges of weight 1 by their betweenness centrality
    edges_with_weight_1_sorted = sorted(edges_with_weight_1, key=lambda edge: edge_betweenness[edge], reverse=True)

    # Determine the number of edges to add to H
    N = NV[i]
    if N > len(edges_with_weight_1):
        N = len(edges_with_weight_1)

    if NV[i]-len(edges_with_weight_1_sorted) < 0:
        edges_to_add = edges_with_weight_1_sorted[:N]
        border_sets.append(edges_to_add)
        nx.set_edge_attributes(GG, {edge: parameter + 1 for edge in edges_to_add}, 'weight')
        edges_sorted_by_betweenness_a = [edge for edge in edges_sorted_by_betweenness_a if edge not in edges_to_add]
    else:
        edges_to_add = edges_with_weight_1_sorted[:N]
        edges_sorted_by_betweenness_a = [edge for edge in edges_sorted_by_betweenness_a if edge not in edges_to_add]
        edges_to_add_1 = edges_sorted_by_betweenness_a[:NV[i]-len(edges_with_weight_1_sorted)]
        edges_to_add = edges_to_add + edges_to_add_1
        edges_sorted_by_betweenness_a = [edge for edge in edges_sorted_by_betweenness_a if edge not in edges_to_add_1]
        border_sets.append(edges_to_add)
        nx.set_edge_attributes(GG, {edge: parameter + 1 for edge in edges_to_add}, 'weight')
        
GG = G.copy()
closeness_values = [closeness_function(GG)]
degree_values = [degree_function(GG)]
diameter_values = [diameter_function(GG, parameter=parameter+1)]
for i in range(20):
    for u, v in border_sets[i]:
        GG[u][v]['weight'] = parameter + 1
    diameter_values.append(diameter_function(GG, parameter=parameter+1))
    degree_values.append(degree_function(GG))
    closeness_values.append(closeness_function(GG))

#----------------------------------------------------------------------PLOT RESULTS
vec_aux = []
for i in range(len(degree_values)):
    vec_aux.append([degree_values[i], closeness_values[i], diameter_values[i]])
vec_aux = np.array(vec_aux)
vec_aux = vec_aux - vec_aux[0]
x1 = np.linspace(0, 100, 21)

fig, ax = plt.subplots(figsize=(8, 4))
markers = ["," , "o" , "^" ]
colors = ['#000000','#00FFFF', '#FF4500']
colors_edge = ['b','c', 'y']
labels = ['$s_{\mathcal{G}}$ ','$c_{\mathcal{G}}$','$d_{\mathcal{G}}$']
for i in range(3):
    plt.plot(x1, vec_aux[:, i],
                color=colors[i],
                markeredgecolor=colors[i],
                lw=2,
                ls='--',
                markerfacecolor=colors_edge[i],
                marker='o',
                #edgecolor=colors_edge[i],
                label=labels[i],
                markersize=7)

# Set the x and y limits
ax.set_xlim([min(x1), max(x1)])
ax.set_ylim([min(vec_aux.flatten()), max(vec_aux.flatten())])


# Add a horizontal colorbar
cbar = fig.colorbar(cm.ScalarMappable(cmap=plt.cm.get_cmap('tab20', 20)), ax=ax, orientation='horizontal', aspect=50, format='')
cbar.ax.tick_params(which='both', length=0, pad=5, labelcolor='black', labelsize=12)

# Set the x and y labels
ax.xaxis.set_visible(False)
ax.set_ylabel(r'Gradual improvement of $J$', fontsize=12)

# Set the legend
ax.legend(loc='upper left', fontsize=12)
budget_x = 250*100/M  # replace with the x-position of your budget FROM 3 TO 4
budget_label = 'Budget'

# Plot the vertical line
plt.axvline(x=budget_x, color='r', linestyle='--')

# Add the label
plt.text(budget_x, ax.get_ylim()[1], budget_label, color='r', va='bottom', ha='center')

# Save and show the plot
plt.savefig('./Images/greedy_from_third_Diameter_1_to_2.png', dpi=300)
plt.show()
