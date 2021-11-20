# # import cv2, numpy as np
# # import matplotlib.pylab as plt
# #
# # img1 = cv2.imread('./photo/0_1.png')
# # img2 = cv2.imread('./photo/1_1.png')
# # img3 = cv2.imread('./photo/2_1.png')
# # img4 = cv2.imread('./photo/3_1.png')
# #
# # cv2.imshow('query', img1)
# # imgs = [img1, img2, img3, img4]
# # hists = []
# # for i, img in enumerate(imgs) :
# #     plt.subplot(1,len(imgs),i+1)
# #     plt.title('img%d'% (i+1))
# #     plt.axis('off')
# #     plt.imshow(img[:,:,::-1])
# #     #---① 각 이미지를 HSV로 변환
# #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #     #---② H,S 채널에 대한 히스토그램 계산
# #     hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
# #     #---③ 0~1로 정규화
# #     cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
# #     hists.append(hist)
# #
# #
# # query = hists[0]
# # methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR,
# #            'INTERSECT':cv2.HISTCMP_INTERSECT,
# #            'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}
# # for j, (name, flag) in enumerate(methods.items()):
# #     print('%-10s'%name, end='\t')
# #     for i, (hist, img) in enumerate(zip(hists, imgs)):
# #         #---④ 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
# #         ret = cv2.compareHist(query, hist, flag)
# #         if flag == cv2.HISTCMP_INTERSECT: #교차 분석인 경우
# #             ret = ret/np.sum(query)        #비교대상으로 나누어 1로 정규화
# #         print("img%d:%7.2f"% (i+1 , ret), end='\t')
# #     print()
# # plt.show()
#
# # import cv2
# # import numpy as np
# #
# # # 이미지 불러오기
# # img = cv2.imread('./photo/4.png')
# #
# # # 변환 graky
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# # # 임계값 조절
# # mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
# #
# # # mask
# # mask = 255 - mask
# #
# # # morphology 적용
# # # borderconstant 사용
# # kernel = np.ones((3,3), np.uint8)
# # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# #
# # # anti-alias the mask
# # # blur alpha channel
# # mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
# #
# # # linear stretch so that 127.5 goes to 0, but 255 stays 255
# # mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
# #
# # # put mask into alpha channel
# # result = img.copy()
# # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# # result[:, :, 3] = mask
# #
# # # 저장
# # cv2.imwrite('./photo/4_1.png', result)
#
# import numpy as np
# import cv2
#
# img_color = cv2.imread('./photo/8888.png') # 이미지 파일을 컬러로 불러옴
# height, width = img_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]
#
# img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환
#
# lower_blue = (0-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
# upper_blue = (0+10, 255, 255)
# # lower_blue = (150, 50, 50) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
# # upper_blue = (180, 255, 255)
# # lower_red = np.array([150, 50, 50]) # 빨강색 범위 upper_red = np.array([180, 255, 255])
# # upper_red = np.array([180, 255, 255])
# print('t')
# img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색

# img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)
# 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득

# cv2.imwrite('./photo/4_1.png', img_mask)
# cv2.imshow('img_color', img_color)
# cv2.imshow('img_mask', img_mask)
# # cv2.imshow('img_color', img_result)
#
# cv2.waitKey(0)
#
# # import numpy as np
# # import cv2
# #
# #
# # imageA = cv2.imread('./photo/999_9.jpeg') # 왼쪽 사진
# # imageB = cv2.imread('./photo/999_8.jpeg') # 오른쪽 사진
# #
# # grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
# # grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
# #
# #
# # sift = cv2.xfeatures2d.SIFT_create()
# # kpA, desA = sift.detectAndCompute(grayA, None)
# # kpB, desB = sift.detectAndCompute(grayB, None)
# #
# #
# # bf = cv2.BFMatcher()
# # matches = bf.match(desA,desB)
# #
# #
# # sorted_matches = sorted(matches, key = lambda x : x.distance)
# # res = cv2.drawMatches(imageA, kpA, imageB, kpB, sorted_matches[:30], None, flags = 2)
# #
# # cv2.imshow('res', res)
# import cv2, numpy as np
# from tqdm import tqdm
#
# a = cv2.imread('./photo/0_1.png')
# b = cv2.imread('./photo/1_1.png')
#
# n=0
# s=0
# print(len(a[0]))
# print(b[0])
# # print(np.array_equal(a[0][0],b[0][0]))
# # for i in tqdm(range(len(a))):
# #     for k in tqdm(range(len(a[i]))):
# #         if np.array_equal(a[i][k], b[i][k]):
# #             n += 1
# #         s += 1
#
# print(n,s)

import cv2
import numpy as np
# import glob
from tqdm import tqdm

# 영상 읽기 및 표시
img = cv2.imread('./photo/2.png')
print(img.shape)
# print('img',img)
# print('img_mask',img_mask)
# img = img_mask

# cv2.imshow('query', img)

# 비교할 영상들이 있는 경로 ---①
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
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance

# 권총 영상의 해쉬 구하기 ---④
query_hash = img2hash(img)

# 데이타 셋 영상 한개 읽어서 표시 ---⑥
img = cv2.imread('./photo/8888.png')
# 데이타 셋 영상 한개의 해시  ---⑦
a_hash = img2hash(img)
# 해밍 거리 산출 ---⑧
dst = hamming_distance(query_hash, a_hash)
if dst/256 < 0.01: # 해밍거리 25% 이내만 출력 ---⑨
    print('./photo/3_1.png', dst/256)
    print('true')
else :
    print('./photo/3_1.png', dst / 256)
    print('false')