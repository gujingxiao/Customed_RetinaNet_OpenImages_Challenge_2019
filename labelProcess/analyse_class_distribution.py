# coding: utf-8
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import pandas as pd
from labelProcess.label_levels import *

def get_children_labels():
    d1, d2 = get_description_for_labels()
    parents = dict()
    for r in d2.keys():
        parents[r] = []

    arr = json.load(open(INPUT_PATH + 'challenge-2019-label500-hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    childList = []
    for i, l in enumerate(lst):
        if 'Subcategory' not in l:
            child = d1[l['LabelName']]
            childList.append(child)
        else:
            sub1 = l['Subcategory']
            for i1, l1 in enumerate(sub1):
                if 'Subcategory' not in l1:
                    child = d1[l1['LabelName']]
                    childList.append(child)
                else:
                    sub2 = l1['Subcategory']
                    for i2, l2 in enumerate(sub2):
                        if 'Subcategory' not in l2:
                            child = d1[l2['LabelName']]
                            childList.append(child)
                        else:
                            sub3 = l2['Subcategory']
                            for i3, l3 in enumerate(sub3):
                                if 'Subcategory' not in l3:
                                    child = d1[l3['LabelName']]
                                    childList.append(child)
                                else:
                                    sub4 = l3['Subcategory']
                                    for i4, l4 in enumerate(sub4):
                                        if 'Subcategory' not in l4:
                                            child = d1[l4['LabelName']]
                                            childList.append(child)
                                        else:
                                            print(l4)

    return childList

# Children
childList = get_children_labels()
d1, d2 = get_description_for_labels()
childLabelList = []
for item in childList:
    childLabelList.append(d2[item])

childList = list(set(childList))
childLabelList=list(set(childLabelList))
# print(len(childList), childList)
# print(len(childLabelList), childLabelList)

# Parents
parentLabelList = []
parentList = []
for idx, key in enumerate(d1):
    if key not in childLabelList:
        parentLabelList.append(key)
        parentList.append(d1[key])
print(parentList)
print(parentLabelList)

csvfile = ROOT_PATH + 'label/challenge-2019-train-detection-bbox.csv'

boxes = pd.read_csv(csvfile)
parentBoxes = boxes[boxes['LabelName'].isin(parentLabelList)]
print(len(boxes), len(parentBoxes))

reduced_boxes = parentBoxes['LabelName']
valueList = reduced_boxes.value_counts()
dict_value = {'label':valueList.index,'number':valueList.values}

# Level Parent Number:1418594 - 27272
labels_to_find = []
for l in list(dict_value['label'][0:20]):
    labels_to_find.append(d1[l])
print(dict_value['label'][0:20])
print(len(labels_to_find), labels_to_find)
print(dict_value['number'][0:20])

# Level Parent Number:1418594 - 27272
labels_to_find = []
for l in list(dict_value['label'][20:]):
    labels_to_find.append(d1[l])
print(dict_value['label'][20:])
print(len(labels_to_find), labels_to_find)
print(dict_value['number'][20:])

# csvfile = ROOT_PATH + 'label/challenge-2019-train-detection-bbox.csv'
#
# boxes = pd.read_csv(csvfile)
# childBoxes = boxes[boxes['LabelName'].isin(childLabelList)]
# print(len(boxes), len(childBoxes))
#
# reduced_boxes = childBoxes['LabelName']
# valueList = reduced_boxes.value_counts()
# dict_value = {'label':valueList.index,'number':valueList.values}
#
# for index in range(12):
#     print(dict_value['label'][index * 40 : (index + 1) * 40])
#
#
# # Level 1 Top 0-50 Number:1418594 - 27272
# labels_to_find = []
# for l in list(dict_value['label'][0:38]):
#     labels_to_find.append(d1[l])
# print(dict_value['label'][0:38])
# print(len(labels_to_find), labels_to_find)
# print(dict_value['number'][0:38])
#
# # Level 2 Top 50-200 Number:26236 - 2388
# labels_to_find = []
# for l in list(dict_value['label'][38:173]):
#     labels_to_find.append(d1[l])
# print(dict_value['label'][38:173])
# print(len(labels_to_find), labels_to_find)
# print(dict_value['number'][38:173])
#
# # Level 3 Top 200-350 Number:2384 - 710
# labels_to_find = []
# for l in list(dict_value['label'][173:308]):
#     labels_to_find.append(d1[l])
# print(dict_value['label'][173:308])
# print(len(labels_to_find), labels_to_find)
# print(dict_value['number'][173:308])
#
# # Level 4 Top 350-500 Number:688 - 14
# labels_to_find = []
# for l in list(dict_value['label'][308:]):
#     labels_to_find.append(d1[l])
# print(dict_value['label'][308:])
# print(len(labels_to_find), labels_to_find)
# print(dict_value['number'][308:])

