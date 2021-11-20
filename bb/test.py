# list = []
# list1 = [1,2,3]
# list2 = [1,2,3]
# list3 = [2,3,4]
# list.append(list1)
# list.append(list2)
# list.append(list3)
# print(list)
# set_list = set(list)
# print(set_list)

import pandas as pd
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim

# df = pd.read_csv('./data/data.csv')
# print(df)


# location_point = (35.89629,128.62200)
# G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk',custom_filter='[“highway”~”motorway|trunk”]')


# ox.save_graph_osm(G, filename='./data/test.osm')      # 지도 세이브



# df = df.append({'time': 60, 'node': list(G)[0], 'like': 0},ignore_index=True)
# df = df.append({'time': 60, 'node': list(G)[1], 'like': 0},ignore_index=True)
# df = df.append({'time': 60, 'node': list(G)[2], 'like': 0},ignore_index=True)
# df = df.append({'time': 60, 'node': list(G)[3], 'like': 0},ignore_index=True)
# df = df.append({'time': 60, 'node': list(G)[4], 'like': 0},ignore_index=True)
#
#
# filt = (df.time == 60) & (df.node == list(G)[4])
# data = df[filt]['like'] + 1
# df.loc[filt, 'like'] = data
# print(df)


# app = Nominatim(user_agent = 'South Korea')
# location = app.geocode('영진 전문대')
# print(location.latitude, location.longitude)
# def geocoding(address):
#     geo = geolocoder.geocode(address)
#     crd = (geo.latitude, geo.longitude)
#     print(crd)
#     return crd


places = ['북구, 대구, 대한민국']
G = ox.graph_from_place(['대구'], network_type='walk', simplify=True)
#
# route = nx.shortest_path(G, list(G)[0], list(G)[3], weight='length')
# fig, ax = ox.plot_graph_route(G, route, node_size=0)