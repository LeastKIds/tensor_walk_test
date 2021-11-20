import networkx as nx
import osmnx as ox
import os.path
import pandas as pd
import random
import SearchPathAlgorithm as al


# 지도 생성
location_point = (35.89629,128.62200)
G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')

# 출발지 도착지 미리 설정
ss_node = (128.62200, 35.89629)
ee_node = (128.61605, 35.89572)

# 전체 점의 수
g_node = len(list(G))

# 데이터 생성하기
if not os.path.isfile('./data/data.csv'):   # 데이터가 없는 맨 처음 실행시 csv파일 생성
    # print(list(G))
    df = pd.DataFrame(columns=['time', 'node', 'like']) # 데이터 프레임에 칼럼 설정
    df.to_csv('./data/data.csv', index=False)    # 칼럼 설정 후 저장.
else:
    df = pd.read_csv('./data/data.csv') # 있다면 데이터 가져오기.
    print(len(df))

data_size = False
if len(df) >= 10000:
    data_size = True

# 텍스트 인터페이스
while True:

    save_route = [] # 경로들 저장
    save_length = [] # 각각 경로들의 점 개수
    save_time = []  # 각각 경로들의 걸리는 시간
    routes = []
    photo = 0   # 얻는 경로의 수 (초기설정)


    print('enter time(exit=0) : ', end='')
    time = int(input())

    # 시간이 0이면 종료
    if time == 0:
        df.to_csv('./data/data.csv', index=False)
        print('save')
        break

    # 경로 찾기 시작
    while True:
        if photo >= 3:  # 경로 3개를 찾으면 종료 조건으로 들어감
            check_route = set(list(map(tuple, save_route))) # 찾은 경로들을 튜플로 전환
            if len(check_route) == len(save_route): # 튜플로 전환했을 시, 중복되는 경로가 있으면 변경전과 길이가 다르다.
                break
            else:   # 길이가 다르기 때문에 중복되는 길이 있으므로, 다시 경로 탐색
                print('duplication!!')
                photo -= 1  # 중복이 됬으므로 해당 경로는 무효 처리

        if not data_size:   # 초기 데이터가 부족할 경우 데이터 수집 목적으로 랜덤 경로 설정
            g_node_range = random.randrange(1, 8)   # 추가할 점의 개수를 랜덤으로 적용

            # 위에서 뽑은 랜덤한 점의 개수 만큼, 랜덤한 점을 뽑음
            route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
            # route 함수에 넣어서 해당 결과 값이 성공인지 아닌지 확인
            length_sum, time_sum, success, route_sum = al.route(route_list, G, ss_node, ee_node, time)

            if success:     #만약 성공시
                set_route = set(route_sum)  # 너무 중복된 길인지 확인
                print('len(set_route : ', end='')
                print(len(set_route))
                print('len(route_sum) : ', end='')
                print(len(route_sum))
                if len(set_route) / len(route_sum) > 0.85:  # 같은 점들을 제거 했을시 해당 수치를 넘어야 함
                    # 해당 경로를 나중에 볼 수 있도록 저장
                    save_route.append(route_sum)
                    save_length.append(length_sum)
                    save_time.append(time_sum)
                    # 경로가 추가 됬으므로 변수 하나 올리기
                    photo += 1

    # 해당 경로들을 보여줌
    for i in range(len(save_route)):
        fig= ox.plot_graph_route(G, save_route[i], node_size=0)

    # 경로 선택
    print('--------------------------------')
    print('select map(again : 0) : ', end='')
    select = int(input())

    if select == 0:
        continue

    # 해당 경로를 따로 변수로 저장
    select_route = save_route[select-1]
    # 해당 경로에 있는 점들을 데이터화 해 저장 (시작점과 도착점은 제거)
    for i in range(1,len(select_route)-1):
        # 해당 하는 시간이 일치하고, 점이 일치하는지 찾기 위한 필터
        filt = (df.time == time) & (df.node == select_route[i])
        if len(df[filt]) == 0:  # 만약 일치하는 데이터가 없다면(아직 데이터에 없는 점이라면)
            # 판다스로 데이터 프레임에 추가
            df = df.append({'time' : time, 'node' : select_route[i], 'like' : 0},ignore_index=True)
        else: # 만약 일치하는 데이터가 있다면
            point = df[filt]['like'] + 1    # 해당 점이 불린 횟수에서 1을 더 추가해준다.(선호도)
            # 많이 불릴 수록 선호하는 점이기 때문에
            # 불린 횟수를 선호도로 체크

            # 해당 데이터로 바꿔줌
            df.loc[filt, 'like'] = point
