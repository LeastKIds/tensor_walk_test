import os.path
import tensorflow as tf
import cv2
import pandas as pd
import SearchPathAlgorithm as al
import numpy as np
from tqdm import tqdm
import random
import osmnx as ox
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import time as t


#######################################################################################
# 노드 데이터 파일이 있는지 확인
def node_data_create():
    if not os.path.isfile('./data/node_data.csv'):
        df = pd.DataFrame(columns=['time', 'node', 'like', 'start', 'end', 'now'])  # 데이터 프레임에 칼럼 설정
        df.to_csv('./data/node_data.csv', index=False)


########################################################################################
# 데이터 수집
def data_collection(time, location_point, ss_node, ee_node):
    model = load_model('./model')
    print('모델 불러오기')
    df = pd.read_csv('./data/node_data.csv')
    print('cvs 파일 불러오기')
    G = create_map(location_point)
    print('지도 불러오기')
    g_node = len(list(G))
    print('노드 갯수 체크')

    for i in tqdm(range(100)):
        while True:
            g_node_range = random.randrange(1, 8)  # 추가할 점의 개수를 랜덤으로 적용
            print('랜덤 점 뽑기')
            # 위에서 뽑은 랜덤한 점의 개수 만큼, 랜덤한 점을 뽑음
            route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
            length_sum, time_sum, success, route_sum, start_node, end_node = al.route(route_list, G, ss_node, ee_node,
                                                                                      time)
            print('조건에 맞는 경로 찾음')
            if success:  # 만약 성공시
                set_route = set(route_sum)  # 너무 중복된 길인지 확인
                print('불일치도 : ', len(set_route) / len(route_sum))
                if len(set_route) / len(route_sum) > 0.85:  # 같은 점들을 제거 했을시 해당 수치를 넘어야 함

                    # 이미지로 저장하기 위해 fig로 변환
                    fig, ax = ox.plot_graph_route(G, route_sum, node_size=0, route_linewidth=10)

                    # 텐서로 예측하는 구간
                    print('tensor로 예측 시작')
                    img_mask_compare = node_division(fig)
                    img_mask_compare = cv2.resize(img_mask_compare, (64,64))
                    img_array = image.img_to_array(img_mask_compare)
                    img_batch = np.expand_dims(img_array, axis=0)
                    prediction = model.predict(img_batch)
                    print('에측 끝')
                    if int(prediction[0]) == 0:
                        print('정답 찾음!!')
                        for k in route_sum:
                            df = df.append({'time': time, 'node': k, 'start': start_node,
                                            'end': end_node, 'like': 0, 'now' : t.time()}, ignore_index=True)




                        break

    df.to_csv('./data/node_data.csv', index=False)
    print('csv 저장 완료')
#########################################################################################
# 지도 생성

def create_map(location_point):
    G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type='walk')
    return G


########################################################################################
# 경로만 추출하기

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

    return img_mask_compare_0


if __name__ == '__main__':

    node_data_create()
    print('노드 데이터 생성')

    time = 60
    location_point = (35.89629, 128.62200)
    ss_node = (128.62200, 35.89629)
    ee_node = (128.61605, 35.89572)
    print('초기 설정')

    data_collection(time, location_point, ss_node, ee_node)
