# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:49:10 2018

@author: Zephyr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import torchvision.transforms as transforms
import glob
import os
import PIL.Image as Image
from model import *
import torch.nn as nn
paths = [
     'fire5'
]
cloud = np.zeros([len(paths)])
# cloud[5] = 1

input_height = 224
input_width = 224
input_mean = 0
input_std = 255
threshold_alarm = 0.85
red = (0, 0, 255)
green = (0, 255, 0)
threshold_daytime = 100
show_time = False

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = TridentNetwork()
model.load_state_dict(torch.load('./Storage_model/TridentNetwork'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

results = np.zeros((45, 2))
is_fire = {0: 'Fire', 1: 'No Fire'}
for pic in range(len(paths)):
    firenum = 0
    videoName = paths[pic]
    framesNum = len(os.listdir("./generate_video/image/" + videoName + "/original_image"))
    for a in range(framesNum):
        fireflag = False
        imgDir = glob.glob("./generate_video/image/" + videoName + "/original_image/*.jpg")[a]
        img = cv2.resize(cv2.imread(imgDir), (1920, 1080))
        # img = Image.open(imgDir)
        # if show_time:
        #     region_mean = img[1:300, 1001:1300, :].mean()
        # ----------------------
        # data_in = transform(img).resize(1, 3, 224, 224)
        # data_out = model(data_in)
        # out_P = data_out.exp() / data_out.exp().sum()
        # print(out_P)
        # _, pre = torch.max(data_out, 1)
        # print('Condition:', is_fire[pre.item()])
        # ------------------------
        for h in range(5):
            for w in range(9):
                sub_image = img[h * 180:h * 180 + 359, w * 180 + 60:w * 180 + 359 + 60, :]
                data_in = transform(Image.fromarray(sub_image)).resize(1, 3, 224, 224)
                data_out = model(data_in)
                # output the Probability of the occurrence of wildfire
                # out_P = data_out.exp() / data_out.exp().sum()
                results[h * 9 + w, :] = data_out.numpy(force=True)
        # input = transform(img)
        # output = model(input)
        for w in range(7):
            cv2.line(img, (60, w * 180), (1860, w * 180), green, 2)
        for h in range(11):
            cv2.line(img, (h * 180 + 60, 0), (h * 180 + 60, 1080), green, 2)
        for h in range(5):
            for w in range(9):
                if h < cloud[pic]:
                    cv2.putText(img, 'Off', (w * 180 + 60, h * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
                else:
                    if results[h * 9 + w][0] > threshold_alarm:
                        color = red
                        cv2.rectangle(img, (w * 180 + 60, h * 180), (w * 180 + 360 + 60, h * 180 + 360), red, 2)
                        fireflag = True
                    else:
                        color = green
                    cv2.putText(img, str(results[h * 9 + w][0] * 100)[0:5] + "%", (w * 180 + 60, h * 180 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # if show_time:
        #     if region_mean > threshold_daytime:
        #         cv2.putText(img, str(region_mean)[0:5] + "daytime", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        #     else:
        #         cv2.putText(img, str(region_mean)[0:5] + "night", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        cv2.imwrite("./generate_video/image/" + videoName + "/result/" + str(a + 1) + '.jpg', img)

        print('Dataset:{},Number of image:{}'.format(videoName, str(a)))
        # tf.reset_default_graph()
        if fireflag:
            firenum += 1

