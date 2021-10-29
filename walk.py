# import networkx as nx
# import osmnx as ox
# import requests
# import matplotlib.cm as cm
# import matplotlib.colors as colors
#
# ox.config(use_cache=True, log_console=True)
# ox.__version__
#
# G = ox.graph_from_place('북구, 대구, 대한민국', network_type='walk')
# fig, ax = ox.plot_graph(G, node_color='r')
import networkx as nx
import osmnx as ox
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import pandas as pd

from PIL import Image, ImageOps, ImageColor, ImageFont, ImageDraw

print(f"The NetworkX package is version {nx.__version__}")
print(f"The OSMNX package is version {ox.__version__}")
print(f"The Request package is version {requests.__version__}")
# print(f"The PIL package is version {PIL.__version__}")

###############################################################################
#                                3. Get Data                                  #
###############################################################################

places = ['북구, 대구, 대한민국']
G = ox.graph_from_place(places, network_type='all', simplify=True)

###############################################################################
#                               4. Unpack Data                                #
###############################################################################

u = []
v = []
key = []
data = []
for uu, vv, kkey, ddata in G.edges(keys=True, data=True):
    v.append(uu)
    v.append(vv)
    key.append(key)
    data.append(ddata)

# print(data[0])
# print(data[1])
# print(data[10]['name'])

time_pd = pd.DataFrame(data)
time_pd.to_csv('data.csv',mode='w',encoding='utf-8-sig')


# ###############################################################################
# #                5. Assign Each Segment a Color Based on its Length           #
# ###############################################################################
# # List to store colors
#
# roadColors = []
#
# # The length is in meters
# for item in data:
#     if "length" in item.keys():
#
#         if item["length"] <= 100:
#             color = "#d40a47"
#
#         elif item["length"] > 100 and item["length"] <= 200:
#             color = "#e78119"
#
#         elif item["length"] > 200 and item["length"] <= 400:
#             color = "#30bab0"
#
#         elif item["length"] > 400 and item["length"] <= 800:
#             color = "#bbbbbb"
#
#         else:
#             color = "w"
#
#     roadColors.append(color)
#
# ###############################################################################
# #                6. Assign Each Segment a Width Based on its type             #
# ###############################################################################
# # List to store linewidths
#
# roadWidths = []
#
# for item in data:
#     if "footway" in item["highway"]:
#         linewidth = 1
#
#     else:
#         linewidth = 2.5
#
#     roadWidths.append(linewidth)
#
# ###############################################################################
# #                                 7. Make Map                                 #
# ###############################################################################
# # Center of map
# latitude = 35.923364943153565
# longitude = 128.5764782484241
#
# # Bbox sides
# north = latitude + 0.05
# south = latitude - 0.05
# east = longitude + 0.08
# west = longitude - 0.08
#
# # Make Map
# fig, ax = ox.plot_graph(G, node_size=0, bbox=(north, south, east, west),  dpi=600, bgcolor="#061529",
#                         save=False, edge_color=roadColors,
#                         edge_linewidth=roadWidths, edge_alpha=1)
#
# # Text and marker size
# markersize = 5
# fontsize = 5
#
# # Add legend
# legend_elements = [Line2D([0], [0], marker='s', color="#061529", label='Length < 100 m',
#                           markerfacecolor="#d40a47", markersize=markersize),
#
#                    Line2D([0], [0], marker='s', color="#061529", label='Length between 100-200 m',
#                           markerfacecolor="#e78119", markersize=markersize),
#
#                    Line2D([0], [0], marker='s', color="#061529", label='Length between 200-400 m',
#                           markerfacecolor="#30bab0", markersize=markersize),
#
#                    Line2D([0], [0], marker='s', color="#061529", label='Length between 400-800 m',
#                           markerfacecolor="#bbbbbb", markersize=markersize),
#
#                    Line2D([0], [0], marker='s', color="#061529", label='Length > 800 m',
#                           markerfacecolor="w", markersize=markersize)]
#
# l = ax.legend(handles=legend_elements, bbox_to_anchor=(0.0, 0.0), frameon=True, ncol=1,
#               facecolor='#061529', framealpha=0.9,
#               loc='lower left', fontsize=fontsize, prop={'family': "Georgia", 'size': fontsize})
#
# # Legend font color
# for text in l.get_texts():
#     text.set_color("w")
#
# # Save figure
# fig.savefig("Lawrence.png", dpi=1000, bbox_inches='tight', format="png", facecolor=fig.get_facecolor(), transparent=True)