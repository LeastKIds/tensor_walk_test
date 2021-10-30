import networkx as nx
import osmnx as ox
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import pandas as pd

location_point = (35.89629,128.62200)
G1_2 = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='drive')

print(G1_2.edges[0].data)

# orig_node = ox.get_nearest_node(G, (36.562687, 128.733008)) #안동역
# dest_node = ox.get_nearest_node(G, (36.576043, 128.505541)) #경북도청

# # fig, ax = ox.plot_graph(G1_2,node_edgecolor='r')
# orig = list(G1_2)[0]
# dest = list(G1_2)[15]
# route = ox.shortest_path(G1_2, orig, orig, weight='length')
# fig1_1, ax = ox.plot_graph_route(G1_2, route, route_color='y', route_linewidth=6, node_size=0.5)

# fig.savefig("./simpleMap/Daegu.png", dpi=1000, bbox_inches='tight', format="png", facecolor=fig.get_facecolor(), transparent=True)

# 유전 알고리즘( 여러 경로를 거쳐 최단 거리 )
# Traveling salesman problem 약자로 TSP (문제)