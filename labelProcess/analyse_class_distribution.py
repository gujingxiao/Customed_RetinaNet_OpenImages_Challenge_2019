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

# Level 1 Top 0-50 Number:1418594 - 27272
labels_to_find = []
for l in list(dict_value['label'][0:50]):
    labels_to_find.append(d1[l])
print(dict_value['label'][0:50])
print(labels_to_find)
print(dict_value['number'][0:50])

# Level 2 Top 50-200 Number:26236 - 2388
labels_to_find = []
for l in list(dict_value['label'][50:200]):
    labels_to_find.append(d1[l])
print(dict_value['label'][50:200])
print(labels_to_find)
print(dict_value['number'][50:200])

# Level 3 Top 200-350 Number:2384 - 710
labels_to_find = []
for l in list(dict_value['label'][200:350]):
    labels_to_find.append(d1[l])
print(dict_value['label'][200:350])
print(labels_to_find)
print(dict_value['number'][200:350])

# Level 4 Top 350-500 Number:688 - 14
labels_to_find = []
for l in list(dict_value['label'][350:500]):
    labels_to_find.append(d1[l])
print(dict_value['label'][350:500])
print(labels_to_find)
print(dict_value['number'][350:500])

