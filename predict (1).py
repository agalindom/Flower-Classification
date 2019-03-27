import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn
from torch import optim
import numpy as np
from PIL import Image
import seaborn as sns
import argparse
import m_func

arg = argparse.ArgumentParser(description='predict_a_pic')
arg.add_argument('pic_to_predict', default='/home/workspace/ImageClassifier/flowers/test/100/image_07926.jpg', nargs = '*', action='store', type = str)
arg.add_argument('checkpoint', default = '/home/workspace/chkpoint.pth', nargs = '*', action = 'store', type = str)
arg.add_argument('--topk', default = 5, dest = 'top_k', action = 'store', type = int)
arg.add_argument('--category', dest = 'category_names', action = 'store', default = '/home/workspace/ImageClassifier/cat_to_name.json')

pa = arg.parse_args()
image_path = pa.pic_to_predict
number_of_ps_classes = pa.top_k
trained_model = pa.checkpoint
mapping = pa.category_names

chk_model = m_func.load_checkpoint(PATH = trained_model)
print(chk_model)
# m_func.display_prediction(image_path = image_path, model = chk_model, label_mapping_file = mapping)
with open(mapping, 'r') as json_file:
    cat_to_name = json.load(json_file)

width, height, ratio = m_func.ratio_check(image_path)
ps,classes,flower = m_func.predict(model = chk_model, image_path = image_path, mapping = cat_to_name, width = width, height = height)
prediction = m_func.display_prediction(image_path = image_path, model = chk_model, mapping = cat_to_name, prob = ps, classes = flower)


