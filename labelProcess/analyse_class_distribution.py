# coding: utf-8
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import pandas as pd
from labelProcess.label_levels import ROOT_PATH


def get_description_for_labels():
    out = open(ROOT_PATH + 'label/challenge-2019-classes-description-500.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2

csvfile = ROOT_PATH + 'label/challenge-2019-train-detection-bbox.csv'

boxes = pd.read_csv(csvfile)
print(len(boxes))

reduced_boxes = boxes['LabelName']
valueList = reduced_boxes.value_counts()
dict_value = {'label':valueList.index,'number':valueList.values}

d1, d2 = get_description_for_labels()

# Level 1 Top 0-80 Number:1418594 - 12135
labels_to_find = []
for l in list(dict_value['label'][0:80]):
    labels_to_find.append(d1[l])
print(dict_value['label'][0:80])
print(labels_to_find)
print(dict_value['number'][0:80])

# Level 2 Top 80-200 Number:12053 - 2388
labels_to_find = []
for l in list(dict_value['label'][80:200]):
    labels_to_find.append(d1[l])
print(dict_value['label'][80:200])
print(labels_to_find)
print(dict_value['number'][80:200])

# Level 3 Top 200-400 Number:2384 - 481
labels_to_find = []
for l in list(dict_value['label'][200:400]):
    labels_to_find.append(d1[l])
print(dict_value['label'][200:400])
print(labels_to_find)
print(dict_value['number'][200:400])

# Level 4 Top 400-500 Number:478 - 14
labels_to_find = []
for l in list(dict_value['label'][400:500]):
    labels_to_find.append(d1[l])
print(dict_value['label'][400:500])
print(labels_to_find)
print(dict_value['number'][400:500])

