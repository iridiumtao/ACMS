
import os

import UrlImageRecognizing
import LocalImageRecognizing
import ComputerVision
import kNNorDT

from os import walk

import csv


def run(cv_client, local_path):
    while 1:
        file_name = ""

        # 防止隱藏檔被載入
        for item in os.listdir(path):
            if not item.startswith('.') and os.path.isfile(os.path.join(path, item)):
                print("The loaded file: %s" % item)
                file_name = item

        # 如果有資料就做事
        if file_name:

            # 透過Azure影像辨識API 取得圖片的主要顏色及標籤
            print('Updating image...')
            localCV = LocalImageRecognizing.LocalCV(cv_client, local_path + file_name)
            tag_image = localCV.get_tag_image()
            detect_color = localCV.get_detect_color()

            # 若標籤<10則不執行
            # 機器學習模型的要求
            if len(tag_image) < 10:
                print("Error: The tag of this image (%d) is less than 10." % len(tag_image))
                return

            # 整理標籤格式為10標籤+2顏色
            tmp_list = tag_image[0:10]
            tmp_tmp_list = tmp_list + detect_color
            print(tmp_tmp_list)

            # 將標籤轉為數值
            tmp_list_id = []
            for i in tmp_tmp_list:
                # print(i)
                tmp_list_id.append(kNNorDT.find_index(i))
            print(tmp_list_id)

            # knn預測圖像類型
            knn = kNNorDT.predict(tmp_list_id)
            print("Restaurant category: %d" % knn)

            write_song_uri(knn)

            # 移動已辨識的圖片
            os.replace(local_path+file_name, local_path+"oldP/"+file_name)

            print("Progress Succeed.")

        # 如果沒有資料就說一下沒資料
        else:
            print("Error: Folder is empty.")

        # time.sleep(5)
        break


# 寫入歌曲的Spotify URI至指定的txt文件
def write_song_uri(song_id):
    f = open(r'restaurant-music-code.txt')
    URL = []
    for line in f:
        URL.append(line)

    f = open('demo_song.txt','w')
    f.write(URL[song_id].rstrip())

    f.close()


if __name__ == '__main__':
    computervision_client = ComputerVision.initialize_check()

    path = 'demo_pic/'

    run(computervision_client, path)

    '''
    f = open(r'restaurant-music-code.txt')
    URL = []
    for line in f:
        URL.append(line)

    f = open('demo_song.txt','w')
    f.write(URL[1].rstrip())
    print(URL[1])
    f.close()
    '''

