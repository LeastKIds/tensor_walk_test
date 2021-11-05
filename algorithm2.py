import networkx as nx
import osmnx as ox
import time

def route(list, g, start, end):
    start_node = ox.nearest_nodes(g, start[0],start[1])
    end_node = ox.nearest_nodes(g, end[0], end[1])
    list.insert(0,start_node)
    route_sum=[]
    time_sum=0
    length_sum=0
    for i in range(len(list)-1):
        route = nx.shortest_path(g, list[i], list[i+1], weight='length')
        route_sum.extend(route[1:])
        length = nx.shortest_path_length(g, list[i], list[i+1], weight='length')
        length_sum += length
        time_sum += (length/66.5)
        # if time_sum > time:
        #     return '초과'

    route = nx.shortest_path(g, list[-1], end_node, weight='length')
    route_sum.extend(route[1:])
    fig, ax = ox.plot_graph_route(g, route_sum, node_size=0)
    print(length_sum)
    print(time_sum)


location_point = (35.89629,128.62200)
G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')

start_time = time.time()

list1 = ox.nearest_nodes(G, 128.63568,35.90232)
list2 = ox.nearest_nodes(G, 128.60259,35.92024)
list3 = ox.nearest_nodes(G, 128.59561,35.90047)
list4 = ox.nearest_nodes(G, 128.59561,35.90047)
list5 = ox.nearest_nodes(G, 128.60896,35.90016)
list6 = ox.nearest_nodes(G, 128.61080,35.89523)
list7 = ox.nearest_nodes(G, 128.61500,35.89988)
list8 = ox.nearest_nodes(G, 128.61432,35.89620)
list9 = ox.nearest_nodes(G, 128.61449,35.89398)
list10 = ox.nearest_nodes(G, 128.61234,35.88536)

ss_node = (128.62200, 35.89629)
ee_node = (128.60530, 35.89601)
route_list = [list1]


route(route_list, G, ss_node, ee_node)
print(time.time() - start_time)

# 왔던길 다시 되돌아가는 경우 제거
# route_sum 안에 있는 중복된 노드를 제거(점을 기준으로 하기에 왔던길이 아니라 왔던 골목이 제거됨)
# 1개의 추가된 점 : 4초
# 2개의 추가된 점 : 4초
# 3개의 추가된 점 : 4초
# 10개의 추가된 점 : 6.8초