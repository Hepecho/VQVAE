import csv
import random
import os
import glob
import pandas as pd
import re
from open_clip.zero_shot_metadata import SIMPLE_IMAGENET_TEMPLATES


def get_label(name, id2class):
    pass


root = '../../data/imagenet-10/train'
map_file = '../../data/synset_words.txt'
output_file = '../../data/imagenet-10.csv'
# 对数据地址中的文件夹进行遍历，将类名存放于列表names中
names = os.listdir(root)
# 存放图像地址
images = []

for name in names:
    # glob.glob返回所有匹配的文件路径列表。
    images += glob.glob(os.path.join(root, name, '*.JPEG'))

print(len(images))

# 获取文件名和类别名的映射
id2class = {}
with open(map_file, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        line_list = line.split(' ', 1)
        id2class[line_list[0]] = line_list[1]

print(id2class['n02672831'])

random.seed(0)
class_names = []
templates = SIMPLE_IMAGENET_TEMPLATES

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img', 'caption'])
    for img in images:
    	# 用os.sep切割具有通用性，自动识别路径分隔符windows和linus
        name = img.split(os.sep)[-2]
        labels = id2class[name]  # 包含若干个同义词
        alias_list = labels.split(', ')
        class_names.append(alias_list[0])
        label = random.sample(alias_list, 1)[0]
        template = random.sample(templates, 1)[0]
        # des_label = f'a photo of {label}'
        caption = template(label)
        writer.writerow([img, caption])

class_names = list(set(class_names))

print(class_names)
print(len(class_names))