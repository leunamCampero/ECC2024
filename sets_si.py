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
#bikes.plot(figsize=(10, 10)).set_axis_off()
#bikes = momepy.extend_lines(bikes, 0.001)
#bikes_e = momepy.extend_lines(bikes, 0.0000)
#bikes_extended.plot(figsize=(10, 10)).set_axis_off()
#bikes_e.geometry = momepy.close_gaps(bikes_e, 0.000)
#bikes_e = momepy.remove_false_nodes(bikes)
#bikes = momepy.extend_lines(bikes, 1)
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

parameter = 2

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
print(NV)

GG = G.copy()

# Create the subgraph H
# H = GG.edge_subgraph(edges_with_weight_2)
edges_sorted_by_betweenness_a = edges_sorted_by_betweenness.copy()
border_sets = [] 
# Perform 20 iterations
for i in range(20):
    # Compute the largest connected component of H
    weights_p = [(u, v) for u, v, data in GG.edges(data=True) if data['weight'] >= parameter + 1]
    n1 = len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter])
    # print(len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter]))
    H = GG.edge_subgraph(weights_p)
    largest_cc = max(nx.connected_components(H), key=len)

    # Find the edges of weight 1 that are connected to the largest connected component
    edges_with_weight_1 = [(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter and (u in largest_cc or v in largest_cc)]

    # Sort the edges of weight 1 by their betweenness centrality
    edges_with_weight_1_sorted = sorted(edges_with_weight_1, key=lambda edge: edge_betweenness[edge], reverse=True)

    # Determine the number of edges to add to H
    N = NV[i]
    # print(N)
    if N > len(edges_with_weight_1):
        N = len(edges_with_weight_1)
    # print(N, N-len(edges_with_weight_1))
    # Add the edges to H
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
        # print(len(edges_with_weight_1_sorted[:N]), len(edges_to_add_1), len(e))
        edges_sorted_by_betweenness_a = [edge for edge in edges_sorted_by_betweenness_a if edge not in edges_to_add_1]
        border_sets.append(edges_to_add)
        nx.set_edge_attributes(GG, {edge: parameter + 1 for edge in edges_to_add}, 'weight')
        # edges_sorted_by_betweenness_a = edges_sorted_by_betweenness_a[NV[i]-len(edges_with_weight_1_sorted):]
        # print(NV[i], NV[i]-len(edges_with_weight_1_sorted),len(edges_to_add))
        # print(len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter]))
    # H.add_edges_from(edges_to_add)
    # border_sets.append(edges_to_add)
    # Update the weights of the added edges
    # nx.set_edge_attributes(GG, {edge: parameter + 1 for edge in edges_to_add}, 'weight')
    # n2 = len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter])
    # print(len([(u, v) for u, v, data in GG.edges(data=True) if data['weight'] == parameter]), N, n1-n2, NV[i]-len(edges_with_weight_1_sorted), len(edges_with_weight_1))
    # print(len(edges_to_add))
    # print(len(edges_sorted_by_betweenness_a))
    # print(len(edges_sorted_by_betweenness_a))
    # Remove the added edges from the list of edges sorted by betweenness centrality
    # edges_sorted_by_betweenness_a = edges_sorted_by_betweenness_a[NV[i]:]


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
    print(i)


vec_aux = []
# vec_aux.append([degree_values[0], closeness_values[0], diameter_values[0]])
for i in range(len(degree_values)):
    vec_aux.append([degree_values[i], closeness_values[i], diameter_values[i]])
# print(len(vec_aux))
vec_aux = np.array(vec_aux)
# vec_aux = vec_aux / 100 
# vec_aux = vec_aux / np.linalg.norm(vec_aux, axis=0)
vec_aux = vec_aux - vec_aux[0]
x1 = np.linspace(0, 100, 21)
# print(len(x1))
# Define the colormap plt.cm.get_cmap('tab20', 20)
# edge_color_map = ListedColormap(cm.get_cmap('nipy_spectral_r', 256)(np.linspace(0.05, 0.50, 256)))

# Plot the data
fig, ax = plt.subplots(figsize=(8, 4))
markers = ["," , "o" , "^" ]
colors = ['#000000','#00FFFF', '#FF4500']
colors_edge = ['b','c', 'y']
labels = ['$s_{G}$ ','$c_{G}$','$d_{G}$']
# differences = [diff_1, diff_2, diff_3, diff_4]
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
# for i in range(4):
#     ax.plot(x1, vec_aux[:, i], color='#FF1493', markeredgecolor="#009ACD", lw=2, ls='--', markerfacecolor="#00C957", marker='s', label=["Strength", "Closeness", "Diameter", "Edge Clustering"][i], markersize=7)

# Set the x and y limits
ax.set_xlim([min(x1), max(x1)])
# ax.set_xlim(None)
# ax.set_ylim([min(edge_clustering), max(edge_clustering)])
ax.set_ylim([min(vec_aux.flatten()), max(vec_aux.flatten())])

# mm = [i[1] for i in bc_sums_sorted]
# cbar = fig.colorbar(cm.ScalarMappable(cmap=edge_color_map, norm=plt.Normalize(vmin=min(mm), vmax=max(mm))), ax=ax, orientation='horizontal', aspect=50)
# cbar.ax.tick_params(which='both', length=0, pad=5, labelcolor='black', labelsize=12)
# cbar = fig.colorbar(cm.ScalarMappable(cmap=plt.cm.get_cmap('tab20', 20), norm=plt.Normalize(vmin=min(mm), vmax=max(mm))), ax=ax, orientation='horizontal', aspect=50)
# cbar = fig.colorbar(cm.ScalarMappable(cmap=plt.cm.get_cmap('tab20', 20)), ax=ax, orientation='horizontal', aspect=50)
# cbar.ax.tick_params(which='both', length=0, pad=5, labelcolor='black', labelsize=12)

cbar = fig.colorbar(cm.ScalarMappable(cmap=plt.cm.get_cmap('tab20', 20)), ax=ax, orientation='horizontal', aspect=50, format='')
cbar.ax.tick_params(which='both', length=0, pad=5, labelcolor='black', labelsize=12)

# Add a horizontal colorbar
# cbar = fig.colorbar(cm.ScalarMappable(cmap=edge_color_map, norm=plt.Normalize(vmin=min(set_centralities), vmax=max(set_centralities))), ax=ax, orientation='horizontal', aspect=50)
# cbar.ax.tick_params(which='both', length=0, pad=5, labelcolor='black', labelsize=12)

# Set the x and y labels
# ax.set_xlabel("Percentage of improving roads", fontsize=12)
ax.xaxis.set_visible(False)
ax.set_ylabel(r'Gradual improvement of $\nabla$', fontsize=12)

# Set the legend
ax.legend(loc='upper left', fontsize=12)
budget_x = 250*100/M  # replace with the x-position of your budget FROM 3 TO 4
budget_label = 'Budget'

# Plot the vertical line
plt.axvline(x=budget_x, color='r', linestyle='--')

# Add the label
plt.text(budget_x, ax.get_ylim()[1], budget_label, color='r', va='bottom', ha='center')

# Save and show the plot
plt.savefig('./Images/greedy_from_third_Diameter_2_to_3.png', dpi=300)
# plt.savefig('./Images/greedy_from_third_Diameter_1_to_2.png', dpi=300)
# plt.savefig('./Images/greedy_from_third_Diameter_3_to_4.png', dpi=300)
# plt.savefig('./Images/new_upgrade_test_from_1_to_2_2.png', dpi=800)
# plt.savefig('./Images/new_upgrade_test_from_2_to_3_2.png', dpi=800)
plt.show()
