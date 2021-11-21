from __future__ import absolute_import, division

""" 
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
OS： Ubuntu 18.04
IDE: VS Code 1.39
Language： python == 3.7.4
"""

"""
常用函数
"""

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def makedir(path):
    """根据指定路径创建文件夹"""
    isExists=os.path.exists(path)
    if not isExists:        
        os.mkdir(path)
        return True
    else:
        return False

def read_filename(path):
    """返回指定路径下的文件名称"""
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filenames.append(os.path.join(root, file))
    filenames.sort(key = lambda x:int(x[-10:-4]))
    return filenames

def show_single_image(path, state):
    """显示单帧图像"""
    display_name = 'Display: '
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)
    image = cv.imread(path)
    image_disp = image.copy()
    cv.rectangle(image_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                        (0, 0, 255), 2)
    cv.imshow(display_name, image_disp)
    cv.waitKey(3000)

def show_video(path):
    """顺序播放文件夹中的图片"""
    filenames = read_filename(path)
    for filename in filenames:
        frame = cv.imread(filename)
        cv.imshow('demo', frame)
        print(filename)
        cv.waitKey(30)

def change_filename(path,start_frame=0):
    """修改文件夹中文件命名的位数,start_frame表示新的起点帧"""
    filenames = read_filename(path)
    for filename in filenames:
        new_name = path + '/' + '%06d'%(int(filename.split('/')[-1].split('.jpg')[0])-start_frame) + '.jpg'
        os.rename(filename, new_name)
    