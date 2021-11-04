import networkx as nx
import osmnx as ox
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


location_point = (35.89629,128.62200)   # 지도 한 가운데 위치
# # 지도 세팅, dist : 한 가운데를 기준으로 지도를 그릴 범위, network_type : 지도에 표시할 거리(all, drive, walk　등등)
G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')
# place = 'Piedmont, California, USA'
# G = ox.graph_from_place(place, network_type='drive')

# orig_node = ox.get_nearest_node(G, (36.562687, 128.733008)) #안동역, 해당 위도,경도에서 가장 가까운 점을 찍어줌
# dest_node = ox.get_nearest_node(G, (36.576043, 128.505541)) #경북도청

# 위에서 세팅한 지도를 바탕으로 그림(G1＿2), node_edgecolor : 점의 색 r은 빨간색
# fig, ax = ox.plot_graph(G1_2,node_edgecolor='r')

# G = ox.add_edge_speeds(G)
# G = ox.add_edge_travel_times(G)

# edges = ox.graph_to_gdfs(G, nodes=False)
# edges['highway'] = edges['highway'].astype(str)
# mean_num = edges.groupby('highway')[['length','speed_kph','travel_time']].mean().round(1)
#
# orig = list(G)[1]
# dest = list(G)[120]
#
# route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
#
# fig2, ax = ox.plot_graph_routes(G, routes=[route2], route_colors=['r', 'y'],
#                                route_linewidth=6, node_size=0)

# G = ox.graph_from_place('북구, 대구, 대한민국', network_type='walk', simplify=False)

# fig, ax = ox.plot_graph(G,  node_size=0, edge_linewidth=0.5)
orig_node = ox.nearest_nodes(G, 128.62200,35.89629)
dest_node = ox.nearest_nodes(G, 128.60530, 35.89601)

route = nx.shortest_path(G, orig_node, dest_node, weight='length')
print(route)

dest_node1 = ox.nearest_nodes(G, 128.60496, 35.89144)
route1 = nx.shortest_path(G, dest_node, dest_node1, weight='length')
route.extend(route1[1:])
print(route)
# route1 = nx.shortest_path(G, route, dest_node1, weight='length')
fig, ax = ox.plot_graph_route(G, route, node_size=0)

# 시속 4km -> 1분에 66.5미터
len = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
print(len/66.5)
