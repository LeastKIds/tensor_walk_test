# 필요한 라이브러리
import networkx as nx
import osmnx as ox



location_point = (35.89629,128.62200)   # 지도 한 가운데 위치
# 지도 세팅, dist : 한 가운데를 기준으로 지도를 그릴 범위, network_type : 지도에 표시할 거리(all, drive, walk　등등)
G1_2 = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='drive')


# orig_node = ox.get_nearest_node(G, (36.562687, 128.733008)) #안동역, 해당 위도,경도에서 가장 가까운 점을 찍어줌
# dest_node = ox.get_nearest_node(G, (36.576043, 128.505541)) #경북도청

# 위에서 세팅한 지도를 바탕으로 그림(G1＿2), node_edgecolor : 점의 색 r은 빨간색
fig, ax = ox.plot_graph(G1_2,node_edgecolor='r')

# 최단 거리
# route = ox.shortest_path(G1_2, orig, orig, weight='length')
# 최단거리를 지도에 표시
# fig1_1, ax = ox.plot_graph_route(G1_2, route, route_color='y', route_linewidth=6, node_size=0.5)

# 해당 지도를 사진으로 저장
# fig.savefig("./simpleMap/Daegu.png", dpi=1000, bbox_inches='tight', format="png", facecolor=fig.get_facecolor(), transparent=True)


# 참고해볼 알고리즘
# 유전 알고리즘( 여러 경로를 거쳐 최단 거리 )
# Traveling salesman problem 약자로 TSP (문제)