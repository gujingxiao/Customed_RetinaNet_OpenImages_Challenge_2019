# coding: utf-8
__modified_author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

from labelProcess.label_levels import *
import os
import pandas as pd

def get_empty_df(negative_samples):
    neg_samp = pd.DataFrame(negative_samples, columns=['ImageID'])
    neg_samp['Source'] = 'freeform'
    neg_samp['LabelName'] = ''
    neg_samp['Confidence'] = 1.0
    neg_samp['XMin'] = ''
    neg_samp['XMax'] = ''
    neg_samp['YMin'] = ''
    neg_samp['YMax'] = ''
    neg_samp['IsOccluded'] = 0
    neg_samp['IsTruncated'] = 0
    neg_samp['IsGroupOf'] = 0
    neg_samp['IsDepiction'] = 0
    neg_samp['IsInside'] = 0
    return neg_samp

def create_level1_files(boxes):
    out_dir = OUTPUT_PATH + 'level_1_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-1.csv', 'w')
    for l in LEVEL_1_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-1.csv', index=False)

def create_level2_files(boxes):
    out_dir = OUTPUT_PATH + 'level_2_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-2.csv', 'w')
    for l in LEVEL_2_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-2.csv', index=False)

def create_level3_files(boxes):
    out_dir = OUTPUT_PATH + 'level_3_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-3.csv', 'w')
    for l in LEVEL_3_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-3.csv', index=False)

def create_level4_files(boxes):
    out_dir = OUTPUT_PATH + 'level_4_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-4.csv', 'w')
    for l in LEVEL_4_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-4.csv', index=False)

def create_level5_files(boxes):
    out_dir = OUTPUT_PATH + 'level_5_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-5.csv', 'w')
    for l in LEVEL_5_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-5.csv', index=False)

def create_level6_files(boxes):
    out_dir = OUTPUT_PATH + 'level_6_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-6.csv', 'w')
    for l in LEVEL_6_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-6.csv', index=False)

def create_level7_files(boxes):
    out_dir = OUTPUT_PATH + 'level_7_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-7.csv', 'w')
    for l in LEVEL_7_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
    print(len(reduced_boxes))
    reduced_boxes = pd.concat([reduced_boxes], axis=0)
    reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-7.csv', index=False)

if __name__ == '__main__':
    boxes = pd.read_csv(ROOT_PATH + 'label/challenge-2019-train-detection-bbox.csv')
    print(len(boxes))
    # Remove Group Of boxes!
    boxes = boxes[boxes['IsGroupOf'] == 0].copy()
    print(len(boxes))

    # create_level1_files(boxes)
    # create_level2_files(boxes)
    # create_level3_files(boxes)
    # create_level4_files(boxes)
    # create_level5_files(boxes)
    create_level6_files(boxes)
    create_level7_files(boxes)
