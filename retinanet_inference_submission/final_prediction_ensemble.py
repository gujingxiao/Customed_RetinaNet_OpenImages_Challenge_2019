# coding: utf-8
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import pandas as pd
import time
from retinanet_inference_submission.ensemble_boxes_functions import *

def create_csv_final_ensemble_predictions(input_dirs, out_file, limit_boxes=300, type='avg'):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')

    index = 0
    id_list = []
    preds_list = []
    # Collect all predictions
    for csv in input_dirs:
        subcsv = pd.read_csv(csv)
        ids = subcsv['ImageId'].values
        preds = subcsv['PredictionString'].values
        id_list.append(ids)
        preds_list.append(preds)
        print("Finish reading: ", csv)

    start = time.clock()
    for idx in range(len(id_list[0])):
        id = id_list[0][idx]

        boxes_list = []
        for i in range(len(input_dirs)):
            preds = preds_list[i][idx]
            if str(preds) == 'nan':
                continue

            arr = preds.strip().split(' ')
            if len(arr) % 6 != 0:
                print('Some problem here! {}'.format(id))
                exit()
            boxes = []
            for j in range(0, len(arr), 6):
                part = arr[j:j + 6]
                boxes.append([part[0], float(part[1]), float(part[2]), float(part[3]), float(part[4]), float(part[5])])
            boxes_list.append(boxes)

        if len(boxes_list) < 1:
            out.write(id + ',')
            out.write('\n')
            continue

        merged_boxes = merge_all_boxes_for_image(boxes_list, 0.55, type)
        # print(id, len(filtered_boxes), len(merged_boxes))
        # if len(merged_boxes) > limit_boxes:
        #     # sort by score
        #     print("Large predictions:", len(merged_boxes))
        #     merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        out.write(id + ',')
        for i in range(len(merged_boxes)):
            label = merged_boxes[i][0]
            score = merged_boxes[i][1]
            b = merged_boxes[i][2:]

            str1 = "{} {} {} {} {} {} ".format(label, score, b[0], b[1], b[2], b[3])
            out.write(str1)

        out.write('\n')
        if index % 100 == 0:
            elapsed = (time.clock() - start)
            print(index, '/', len(id_list[0]), "Time used:", elapsed)
        index += 1

    out.close()


def combine_final_parts(input_dirs, out_file):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')

    index = 0
    # Collect all predictions
    for csv in input_dirs:
        subcsv = pd.read_csv(csv)
        ids = subcsv['ImageId'].values
        preds = subcsv['PredictionString'].values

        for idx in range(len(ids)):
            index += 1
            id = ids[idx]
            pred = preds[idx]
            if str(pred) == 'nan':
                out.write(id + ',')
                out.write('\n')
            else:
                out.write(id + ',')
                out.write(pred)
                out.write('\n')

        print("Finish writing: ", csv, index)
    out.close()

def delete_low_scores(input_dirs, thres=0.1):
    out = open('retinanet_final_ensemble_submissions_selected_{}_type_avg.csv'.format(thres), 'w')
    out.write('ImageId,PredictionString\n')

    subcsv = pd.read_csv(input_dirs)
    ids = subcsv['ImageId'].values
    preds = subcsv['PredictionString'].values
    start = time.clock()
    for idx in range(len(ids)):
        id = ids[idx]
        pred = preds[idx]
        out.write(id + ',')
        if str(pred) == 'nan':
            out.write('\n')
        else:
            arr = pred.strip().split(' ')
            if len(arr) % 6 != 0:
                print('Some problem here! {}'.format(id))
                exit()
            for j in range(0, len(arr), 6):
                part = arr[j:j + 6]
                if float(part[1]) > thres:
                    str1 = "{} {} {} {} {} {} ".format(part[0], part[1], part[2], part[3], part[4], part[5])
                    out.write(str1)
            out.write('\n')
        if idx % 1000 == 0:
            elapsed = (time.clock() - start)
            print(idx, '/', len(ids), "Time used:", elapsed)
    out.close()

if __name__ == '__main__':
    limit_boxes = 2100
    type = 'avg'

    # input_dirs = [
    #     'retinanet_ensemble_submission_0.1_0.55_predictions_all_levels_part_0.csv',
    #     'retinanet_resnet152_101_new_ensemble_submission_0.05_0.55_predictions_all_levels_part_0.csv',
    # ]
    #
    # create_csv_final_ensemble_predictions(input_dirs,'retinanet_final_ensemble_submissions_type_{}__part_0.csv'.format(type),limit_boxes, type=type)


    # input_dirs = [
    #     'retinanet_final_ensemble_submissions_type_avg__part_0.csv',
    #     'retinanet_final_ensemble_submissions_type_avg__part_1.csv',
    #     'retinanet_final_ensemble_submissions_type_avg__part_2.csv',
    #     'retinanet_final_ensemble_submissions_type_avg__part_3.csv',
    #     'retinanet_final_ensemble_submissions_type_avg__part_4.csv',
    # ]
    #
    # combine_final_parts(input_dirs, 'retinanet_final_ensemble_submissions_0.05_0.55_type_avg.csv')


    delete_low_scores('retinanet_final_ensemble_submissions.csv', 0.02)