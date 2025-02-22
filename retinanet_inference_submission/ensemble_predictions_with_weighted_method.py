# coding: utf-8
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import os
import glob

import time
from labelProcess.label_levels import *
from retinanet_inference_submission.ensemble_boxes_functions import *

def create_csv_for_retinanet_multiple_predictions(input_dirs, out_file, label_arr, skip_box_thr=0.05, intersection_thr=0.5, limit_boxes=300, type='avg'):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')

    d1, d2 = get_description_for_labels()
    files = glob.glob(input_dirs[0] + '*.pkl')
    index = 0
    start = time.clock()
    for f in files:
        id = os.path.basename(f)[:-4]
        # print('Go for ID: {}'.format(id))
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(input_dirs)):
            f1 = input_dirs[i] + id + '.pkl'
            boxes, scores, labels = load_from_file_fast(f1)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        filtered_boxes = filter_boxes_v2(boxes_list, scores_list, labels_list, skip_box_thr)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, intersection_thr, type)
        # print(id, len(filtered_boxes), len(merged_boxes))
        if len(merged_boxes) > limit_boxes:
            # sort by score
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        out.write(id + ',')
        for i in range(len(merged_boxes)):
            label = int(merged_boxes[i][0])
            score = merged_boxes[i][1]
            b = merged_boxes[i][2:]

            google_name = label_arr[label]
            if '/' not in google_name:
                google_name = d2[google_name]

            xmin = b[0]
            if xmin < 0:
                xmin = 0
            if xmin > 1:
                xmin = 1

            xmax = b[2]
            if xmax < 0:
                xmax = 0
            if xmax > 1:
                xmax = 1

            ymin = b[1]
            if ymin < 0:
                ymin = 0
            if ymin > 1:
                ymin = 1

            ymax = b[3]
            if ymax < 0:
                ymax = 0
            if ymax > 1:
                ymax = 1

            if (xmax < xmin):
                print('X min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                exit()

            if (ymax < ymin):
                print('Y min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                exit()

            if abs(xmax - xmin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], xmin, xmax))
                continue

            if abs(ymax - ymin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], ymin, ymax))
                continue

            str1 = "{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ".format(google_name, score, xmin, ymin, xmax, ymax)
            out.write(str1)

        out.write('\n')
        if index % 1000 == 0:
            elapsed = (time.clock() - start)
            print(index, '/', len(files), "Time used:", elapsed)
        index += 1

    out.close()


if __name__ == '__main__':
    skip_box_thr = 0.15
    intersection_thr = 0.55
    label_level = 5
    limit_boxes = 600
    type = 'avg'

    input_dirs = [
        ROOT_PATH + 'new_pkl/cache_retinanet_level_{}_resnet101/'.format(label_level),
        ROOT_PATH + 'new_pkl/cache_retinanet_level_{}_resnet152/'.format(label_level),
    ]

    if label_level == 1:
        labels_arr = LEVEL_1_LABELS
    elif label_level == 2:
        labels_arr = LEVEL_2_LABELS
    elif label_level == 3:
        labels_arr = LEVEL_3_LABELS
    elif label_level == 4:
        labels_arr = LEVEL_4_LABELS
    elif label_level == 5:
        labels_arr = LEVEL_5_LABELS
    else:
        raise ValueError("Label level ({}) is not supported.".format(label_level))

    create_csv_for_retinanet_multiple_predictions(input_dirs,'retinanet_level_{}_ensemble_thr_{}_iou_{}_type_{}.csv'.format(label_level, skip_box_thr, intersection_thr, type),
                                         labels_arr, skip_box_thr, intersection_thr, limit_boxes, type=type)