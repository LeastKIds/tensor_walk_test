import networkx as nx
import osmnx as ox
import os.path
import pandas as pd
import random
import numpy as np
import datetime
from tqdm import tqdm
import cv2
import SearchPathAlgorithm as al


####################################################################################################
# 지도 그리기
# return : G
def create_map(location_point):
    G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')
    return G



####################################################################################################
# 맵 데이터 가져오기
# return : DataFrame(map)
def map_data():
    if not os.path.isfile('./data/map_data.csv'):  # 데이터가 없는 맨 처음 실행시 csv파일 생성
        # print(list(G))
        df = pd.DataFrame(columns=['time', 'node', 'result', 'start', 'end',
                                   'node1', 'node2', 'node3', 'node4', 'node5', 'node6',
                                   'node7', 'node8',
                                   'real_time', 'now'])  # 데이터 프레임에 칼럼 설정
        df.to_csv('./data/map_data.csv', index=False)  # 칼럼 설정 후 저장.
        return df
    else:
        df = pd.read_csv('./data/map_data.csv')  # 있다면 데이터 가져오기.
        print('데이터의 수 : ',len(df))
        return df
#####################################################################################################
# 노드 데이터 가져오기
# return : DataFrame(node)
def node_data():
    if not os.path.isfile('./data/node_data.csv'):  # 데이터가 없는 맨 처음 실행시 csv파일 생성
        # print(list(G))
        df = pd.DataFrame(columns=['time','node','like','start','end'])  # 데이터 프레임에 칼럼 설정
        df.to_csv('./data/node_data.csv', index=False)  # 칼럼 설정 후 저장.
        return df
    else:
        df = pd.read_csv('./data/node_data.csv')  # 있다면 데이터 가져오기.
        print('데이터의 수 : ',len(df))
        return df


#####################################################################################################
# 맵 선택지 찾기(3개)
# return : save_route, save_length, save_time : 루트들이 담겨 있는 리스트, 루트의 길이가 담겨 있느 리스트, 루트의 시간이 담겨 있는 리스트

def search_map(save_route, save_length, save_time, ss_node, ee_node) :
    photo = 0  # 얻는 경로의 수 (초기설정)
    while True:
        print('select route')
        if photo >= 3:  # 경로 3개를 찾으면 종료 조건으로 들어감
            check_route = set(list(map(tuple, save_route)))  # 찾은 경로들을 튜플로 전환
            if len(check_route) == len(save_route):  # 튜플로 전환했을 시, 중복되는 경로가 있으면 변경전과 길이가 다르다.
                break
            else:  # 길이가 다르기 때문에 중복되는 길이 있으므로, 다시 경로 탐색
                print('duplication!!')
                photo -= 1  # 중복이 됬으므로 해당 경로는 무효 처리

        # if not data_size:  # 초기 데이터가 부족할 경우 데이터 수집 목적으로 랜덤 경로 설정
        g_node_range = random.randrange(1, 8)  # 추가할 점의 개수를 랜덤으로 적용
        # g_node_range = 8
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

    return save_route, save_length, save_time


######################################################################################################
# 복잡한 사진으로 비교하면 두 이미지 유사도가 엉망이 되기 때문에
# 비교하기 쉽도록 경로만 있는 이미지로 바꿈
# return : img_mask_compare_0 : 경로만 추출된 이미지 변수

def node_division(fig):
    # 맵 데이터인 fig를 이미지화 시킴
    # 그러면서 빨간색 경로가 파랑색으로 바뀜
    img_color = np.array(fig.canvas.renderer._renderer)

    height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

    # 추출할 파랑색의 범위를 지정하는 부분
    lower_blue = (120 - 10, 30, 30)  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (120 + 10, 255, 255)

    img = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 범위내의 픽셀들은 흰색, 나머지 검은색

    # 변환된 mask 이미지를 보통 이미지로 전환
    img_mask_compare_0 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 나중에 이미지 학습시키기 위한 정답 데이터로 저장
    cv2.imwrite('./PhotoData/answer/0.png', img_mask_compare_0)

    return img_mask_compare_0


#####################################################################################################
#




######################################################################################################
if __name__ == '__main__':
    # 지도 중심 값 입력
    location_point = (35.89629, 128.62200)

    # 지도 만들기
    G = create_map(location_point)

    # 지도 데이터 생성하기 (함수) 있으면 불러오기
    df_map = map_data()

    # 노드 데이터 생성하기 (함수) 있으면 불러오기
    df_node = node_data()

    # 시간 저장
    global_time = 0

    # 전체 점의 수
    g_node = len(list(G))

    # 첫 번째로 내가 선택한 길 찾기
    while True:
        save_route = []  # 경로들 저장
        save_length = []  # 각각 경로들의 점 개수
        save_time = []  # 각각 경로들의 걸리는 시간


        # 산책할 시간 입력
        print('enter time(exit=0) : ', end='')
        time = int(input())

        # 시간이 0이면 데이터 저장하고 종료
        if time == 0:
            df_map.to_csv('./data/map_data.csv', index=False)
            df_node.to_csv('./data/node_data.csv', index=False)
            print('data save')
            break
        # 시간을 반복문 밖에서도 사용할 수 있게 저장
        global_time = time

        # 출발지 도착지 미리 설정
        ss_node = (128.62200, 35.89629)
        ee_node = (128.61605, 35.89572)

        # 선택할 3개의 경로를 생성하는 함수
        save_route, save_length, save_time = search_map(save_route, save_length, save_time, ss_node, ee_node)

        # 선정된 3개의 경로를 지도로 뿌려주는 곳
        for i in range(len(save_route)):
            fig, ax = ox.plot_graph_route(G, save_route[i], node_size=0)

        # 경로 선택
        print('--------------------------------')
        print('select map(again : 0) : ', end='')
        select = int(input())
        # 만약 0을 입력하면 맨 위로 올라가 경로 3개를 선택하는 함수가 다시 실행
        if select == 0:
            continue

        # 선택된 경로를 지도로 표시
        fig, ax = ox.plot_graph_route(G, save_route[select - 1], node_size=0, route_linewidth=10)


        img_mask_compare_0 = node_division(fig)
