# 一. 載入套件
# 資料處理套件
import cv2
import csv
import random
import numpy as np
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt # plt 用於顯示圖片

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils as np_utils

def imgGenFunc(image_size, flip, rotate):
    # generated image
    train_str = "train.csv"
    train_img_dir = "C1-P1_Train/" 
    csvfile = open(train_str)
    reader = csv.reader(csvfile)
    labels_pic = []
    labels_level = []
    for line in reader:
        labels_pic.append(line[0][:len(line[0])-4])
        labels_level.append(line[1])
    csvfile.close() 
    labels_pic.pop(0)
    labels_level.pop(0)
    picnum = len(labels_pic)
    print("芒果圖片數量: ",picnum)

    X = []
    y = []

    # 轉換圖片的標籤
    for i in range(len(labels_level)):
        labels_level[i] = labels_level[i].replace("A","0")
        labels_level[i] = labels_level[i].replace("B","1")
        labels_level[i] = labels_level[i].replace("C","2")

    # 隨機讀取圖片
    a = 0
    items= []
    for a in range(0, picnum):
        items.append(a)
    c = 1
    # 製作訓練用資料集及標籤
    for i in random.sample(items, picnum):
        
        print(c, end="\r", flush=True)
        c += 1
        
        img = cv2.imread(train_img_dir + labels_pic[i] + ".jpg")
        res = cv2.resize(img,(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

        # flip image
        if (flip is 1):
            img_flip0=cv2.flip(res,0) #垂直翻转
            img_flip1=cv2.flip(res,1) #水平翻转
            img_flip2=cv2.flip(res,-1) #水平垂直翻转
            X.append(img_to_array(img_flip0))
            X.append(img_to_array(img_flip1))
            X.append(img_to_array(img_flip2))

        # rotate image
        if (rotate is 1):
            (h, w) = res.shape[:2]
            center = (w // 2, h // 2)
            M_90 = cv2.getRotationMatrix2D(center, 90, 1.0)
            M_270 = cv2.getRotationMatrix2D(center, 270, 1.0)
            rotated_img_90 = cv2.warpAffine(res, M_90, (w, h))
            rotated_img_270 = cv2.warpAffine(res, M_270, (w, h))
            X.append(img_to_array(rotated_img_90))
            X.append(img_to_array(rotated_img_270))
        
        res = img_to_array(res)
        X.append(res)  
        for j in range(1+flip*3+rotate*2):
            y.append(labels_level[i])


    print("x_l: ",len(X))
    print("y_l: ", len(y))

    # 轉換至array的格式
    X = np.array(X)
    y = np.array(y)# 轉換至float的格式
    for i in range(len(X)):
        X[i] = X[i].astype('float32')# 將標籤轉換至float格式
    y = np_utils.to_categorical(y, num_classes = 3)
    
    return X, y