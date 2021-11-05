import networkx as nx
import osmnx as ox
import time
import random

def route(list, g, start, end, time):
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
        print('======')
        print('time_sum : ', end='')
        print(time_sum)
        print('======')
        if time+10 < time_sum:
            success = False
            return length_sum, time_sum, success, route_sum

    route = nx.shortest_path(g, list[-1], end_node, weight='length')
    route_sum.extend(route[1:])
    length = nx.shortest_path_length(g, list[-1], end_node, weight='length')
    length_sum += length
    time_sum += (length / 66.5)

    if time + 10 > time_sum > time - 10:
        success = True
        return length_sum, time_sum, success, route_sum

    else:
        success = False
        return length_sum, time_sum, success, route_sum


location_point = (35.89629,128.62200)
G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')

start_time = time.time()

# list1 = ox.nearest_nodes(G, 128.63568,35.90232)
# list2 = ox.nearest_nodes(G, 128.60259,35.92024)
# list3 = ox.nearest_nodes(G, 128.59561,35.90047)
# list4 = ox.nearest_nodes(G, 128.59561,35.90047)
# list5 = ox.nearest_nodes(G, 128.60896,35.90016)
# list6 = ox.nearest_nodes(G, 128.61080,35.89523)
# list7 = ox.nearest_nodes(G, 128.61500,35.89988)
# list8 = ox.nearest_nodes(G, 128.61432,35.89620)
# list9 = ox.nearest_nodes(G, 128.61449,35.89398)
# list10 = ox.nearest_nodes(G, 128.61234,35.88536)
# list1 = list(G)[0]
# list2 = list(G)[10]
# list3 = list(G)[20]
# list4 = list(G)[30]
# list5 = list(G)[40]
# list6 = list(G)[50]
# list7 = list(G)[60]
# list8 = list(G)[70]
# list9 = list(G)[80]
# list10 = list(G)[90]

ss_node = (128.62200, 35.89629)
ee_node = (128.61605, 35.89572)
# route_list = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]
g_node = len(list(G))

save_route = []
save_length = []
save_time = []
routes = []
photo = 0
print('start')
while True:
    if photo >= 10:
        check_route = set(list(map(tuple, save_route)))
        if len(check_route) == len(save_route):
            break
        else:
            print('duplication!!')
            photo -= 1
    g_node_range = random.randrange(1,8)
    route_list = [list(G)[random.randrange(0,g_node-1)] for i in range(g_node_range)]
    length_sum, time_sum, success, route_sum = route(route_list,G, ss_node, ee_node, 60)
    if success:
        set_route = set(route_sum)
        print('len(set_route : ', end='')
        print(len(set_route))
        print('len(route_sum) : ', end='')
        print(len(route_sum))
        if len(set_route) / len(route_sum) > 0.85:
            save_route.append(route_sum)
            save_length.append(length_sum)
            save_time.append(time_sum)
            photo += 1



fig, ax = ox.plot_graph_route(G, save_route[0], node_size=0)

for i in range(len(save_route)):
    fig, ax = ox.plot_graph_route(G, save_route[i], node_size=0)
    path = './photo/' + str(i) + '.png'
    fig.savefig(path, dpi=1000, bbox_inches='tight', format="png", facecolor=fig.get_facecolor(),
                transparent=True)

print('save_route : ', end='')
print(save_route)
print('save_length : ', end='')
print(save_length)
print('save_time : ', end='')
print(save_time)

# route(route_list, G, ss_node, ee_node)
# print(time.time() - start_time)

# 왔던길 다시 되돌아가는 경우 제거
# route_sum 안에 있는 중복된 노드를 제거(점을 기준으로 하기에 왔던길이 아니라 왔던 골목이 제거됨)

# 지도에 표시하지 않고　시간만 체크했을 시
# 1개의 추가된 점 : 0.08초, 0.072초, 0.091초
# 2개의 추가된 점 : 0.13초, 0.13초, 0.13초
# 3개의 추가된 점 : 0.14초, 0.14초, 0.14초
# 10개의 추가된 점 : 0.3초, 0,31초, 0.33초