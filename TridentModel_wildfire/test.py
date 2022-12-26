import cv2 as cv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import *

model = TridentNetwork()

model.load_state_dict(torch.load('./Storage_model/TridentNetwork'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

model.eval()
geo_class = {0: 'Fire', 1: 'Nofire'}
path = "./test_data"
for img_name in os.listdir(path):
    img_path = path + '/' + img_name
    img = cv.imread(img_path)
    img_in = transform(img)
    prediction = int(model(img_in).argmax(0))
    print('Image:{},  Class:{}'.format(img_name, geo_class[prediction]))
