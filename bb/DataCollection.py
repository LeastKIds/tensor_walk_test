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

start_node = ox.nearest_nodes(G, ss_node[0],ss_node[1])
end_node = ox.nearest_nodes(G, ee_node[0], ee_node[1])

# 전체 점의 수
g_node = len(list(G))

# 데이터 생성하기
if not os.path.isfile('./data/data_collection.csv'):   # 데이터가 없는 맨 처음 실행시 csv파일 생성
    # print(list(G))
    df = pd.DataFrame(columns=['time', 'node', 'like', 'start', 'end', 'real_time']) # 데이터 프레임에 칼럼 설정
    df.to_csv('./data/data_collection.csv', index=False)    # 칼럼 설정 후 저장.
else:
    df = pd.read_csv('./data/data_collection.csv') # 있다면 데이터 가져오기.
    print(len(df))

data_size = False
if len(df) >= 10000:
    data_size = True

# 시간 저장
global_time = 0


# 첫 데이터 등록
while True:
    save_route = []  # 경로들 저장
    save_length = []  # 각각 경로들의 점 개수
    save_time = []  # 각각 경로들의 걸리는 시간
    routes = []
    photo = 0  # 얻는 경로의 수 (초기설정)

    print('enter time(exit=0) : ', end='')
    time = int(input())

    # 시간이 0이면 종료
    if time == 0:
        df.to_csv('./data/data_collection.csv', index=False)
        print('save')
        break
    # 시간을 반복문 밖에서도 사용할 수 있게 저장
    global_time = time

    # 경로 찾기 시작
    while True:
        print('select route')
        if photo >= 3:  # 경로 3개를 찾으면 종료 조건으로 들어감
            check_route = set(list(map(tuple, save_route)))  # 찾은 경로들을 튜플로 전환
            if len(check_route) == len(save_route):  # 튜플로 전환했을 시, 중복되는 경로가 있으면 변경전과 길이가 다르다.
                break
            else:  # 길이가 다르기 때문에 중복되는 길이 있으므로, 다시 경로 탐색
                print('duplication!!')
                photo -= 1  # 중복이 됬으므로 해당 경로는 무효 처리

        if not data_size:  # 초기 데이터가 부족할 경우 데이터 수집 목적으로 랜덤 경로 설정
            g_node_range = random.randrange(1, 8)  # 추가할 점의 개수를 랜덤으로 적용

            # 위에서 뽑은 랜덤한 점의 개수 만큼, 랜덤한 점을 뽑음
            route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
            # route 함수에 넣어서 해당 결과 값이 성공인지 아닌지 확인
            length_sum, time_sum, success, route_sum = al.route(route_list, G, ss_node, ee_node, time)

            if success:  # 만약 성공시
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


    # 마음에 드느 경로 확인
    for i in range(len(save_route)):
        fig, ax = ox.plot_graph_route(G, save_route[i], node_size=0)

    # 경로 선택
    print('--------------------------------')
    print('select map(again : 0) : ', end='')
    select = int(input())

    if select == 0:
        continue

    fig, ax = ox.plot_graph_route(G, save_route[select-1], node_size=0)
    print('해당 사진 저장중-----')
    # 경로 사진 저장
    fig.savefig("./photo/8888.png", dpi=1000, bbox_inches='tight', format="png", facecolor=fig.get_facecolor(),
                transparent=True)

    break

    ###########################################################################################################
    # 이미지 흑백 화
print('img black/white')
import numpy as np
import cv2

img_color = cv2.imread('./photo/8888.png') # 이미지 파일을 컬러로 불러옴
height, width = img_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환

lower_blue = (0-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (0+10, 255, 255)
img_mask_compare_0 = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색
cv2.imwrite('./photo/8888_1.png', img_mask_compare_0)
# 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
print('이미지에서 경로 말고 제거')

############################################################################################################
# 비슷한 경로를 100개 추려서 저장
from tqdm import tqdm
for z in tqdm(range(100)):
    # 이미지 뽑기
    while True:
        save_route = []  # 경로 저장
        save_length = 0  # 각각 경로들의 점 개수
        save_time = 0  # 각각 경로들의 걸리는 시간




        g_node_range = random.randrange(1, 8)  # 추가할 점의 개수를 랜덤으로 적용

        # 위에서 뽑은 랜덤한 점의 개수 만큼, 랜덤한 점을 뽑음
        route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
        # route 함수에 넣어서 해당 결과 값이 성공인지 아닌지 확인
        length_sum, time_sum, success, route_sum = al.route(route_list, G, ss_node, ee_node, time)

        if success:  # 만약 성공시
            set_route = set(route_sum)  # 너무 중복된 길인지 확인
            print('len(set_route : ', end='')
            print(len(set_route))
            print('len(route_sum) : ', end='')
            print(len(route_sum))
            if len(set_route) / len(route_sum) > 0.85:  # 같은 점들을 제거 했을시 해당 수치를 넘어야 함
                # 해당 경로를 나중에 볼 수 있도록 저장

                fig, ax = ox.plot_graph_route(G, route_sum, node_size=0)
                fig.savefig("./photo/9999.png", dpi=1000, bbox_inches='tight', format="png",
                            facecolor=fig.get_facecolor(),
                            transparent=True)

                img_color = cv2.imread('./photo/9999.png')  # 이미지 파일을 컬러로 불러옴
                height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

                img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

                lower_blue = (0 - 10, 30, 30)  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
                upper_blue = (0 + 10, 255, 255)
                img_mask_compare_1 = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 범위내의 픽셀들은 흰색, 나머지 검은색
                # 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
                print('이미지에서 경로 말고 제거')
                cv2.imwrite('./photo/9999_1.png', img_mask_compare_1)

                ############################################################################
                # 경로 비교


                # 영상 읽기 및 표시
                img = cv2.imread('./photo/8888_1.png')
                # cv2.imshow('query', img)
                #
                # # 비교할 영상들이 있는 경로 ---①
                # search_dir = './photo'


                # 이미지를 16x16 크기의 평균 해쉬로 변환 ---②
                def img2hash(img):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (16, 16))
                    avg = gray.mean()
                    bi = 1 * (gray > avg)
                    return bi


                # 해밍거리 측정 함수 ---③
                def hamming_distance(a, b):
                    a = a.reshape(1, -1)
                    b = b.reshape(1, -1)
                    # 같은 자리의 값이 서로 다른 것들의 합
                    distance = (a != b).sum()
                    return distance


                # 권총 영상의 해쉬 구하기 ---④
                query_hash = img2hash(img)

                # 데이타 셋 영상 한개 읽어서 표시 ---⑥
                img = cv2.imread('./photo/9999_1.png')
                # 데이타 셋 영상 한개의 해시  ---⑦
                a_hash = img2hash(img)
                # 해밍 거리 산출 ---⑧
                dst = hamming_distance(query_hash, a_hash)
                if dst / 256 < 0.01:  # 해밍거리 25% 이내만 출력 ---⑨
                    print('비교 경로', dst / 256)
                    print('true')
                    # filt = (df.time == time) & (df.node == route_sum) & (df.start == start_node) & (df.end == end_node)
                    # if len(df[filt]) == 0:  # 만약 일치하는 데이터가 없다면(아직 데이터에 없는 점이라면)
                    #     # 판다스로 데이터 프레임에 추가
                    df = df.append({'time': time, 'node': route_sum, 'like': 0, 'start' : start_node, 'end' : end_node, 'real_time' : time_sum}, ignore_index=True)
                    # point = df[filt]['like'] + 1  # 해당 점이 불린 횟수에서 1을 더 추가해준다.(선호도)
                    # # 많이 불릴 수록 선호하는 점이기 때문에
                    # # 불린 횟수를 선호도로 체크
                    #
                    # # 해당 데이터로 바꿔줌
                    # df.loc[filt, 'like'] = point

                    break
                else:
                    print('false')













    ########################################################################################
    #
    #
    # # 해당 경로에 있는 점들을 데이터화 해 저장 (시작점과 도착점은 제거)
    # for i in range(1,len(select_route)-1):
    #     # 해당 하는 시간이 일치하고, 점이 일치하는지 찾기 위한 필터
    #     filt = (df.time == time) & (df.node == select_route[i])
    #     if len(df[filt]) == 0:  # 만약 일치하는 데이터가 없다면(아직 데이터에 없는 점이라면)
    #         # 판다스로 데이터 프레임에 추가
    #         df = df.append({'time' : time, 'node' : select_route[i], 'like' : 0},ignore_index=True)
    #     else: # 만약 일치하는 데이터가 있다면
    #         point = df[filt]['like'] + 1    # 해당 점이 불린 횟수에서 1을 더 추가해준다.(선호도)
    #         # 많이 불릴 수록 선호하는 점이기 때문에
    #         # 불린 횟수를 선호도로 체크
    #
    #         # 해당 데이터로 바꿔줌
    #         df.loc[filt, 'like'] = point
