# coding: utf-8
__modified_author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import pandas as pd
from labelProcess.label_levels import *
from retinanet_inference_submission.ensemble_boxes_functions import *

def flatten_boxes(boxes):
    s = ''
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            s += str(boxes[i, j]) + ' '
    return s

def delete_higher_level_classes_from_csv(input_subm, out_file):
    subm = pd.read_csv(input_subm)
    ids = subm['ImageId'].values
    preds = subm['PredictionString'].values
    preds_modified = []
    for i in range(len(ids)):
        if i % 1000 == 0:
            print('Index {}, Go for {}'.format(i, ids[i]))
        id = ids[i]
        if str(preds[i]) == 'nan':
            preds_modified.append('')
            continue
        arr = preds[i].strip().split(' ')
        if len(arr) % 6 != 0:
            print('Some problem here! {}'.format(id))
            exit()
        boxes = []
        for j in range(0, len(arr), 6):
            part = arr[j:j + 6]
            if part[0] in LEVEL_CHILDREN:
                boxes.append(part)
        boxes = np.array(boxes)
        box_str = flatten_boxes(boxes)
        preds_modified.append(box_str)
    subm['PredictionString'] = preds_modified
    subm.to_csv(out_file, index=False)


if __name__ == '__main__':
    delete_higher_level_classes_from_csv('retinanet_ensemble_submission_0.1_0.55_predictions_all_levels.csv',
                                         'retinanet_ensemble_submission_0.1_0.55_predictions.csv')