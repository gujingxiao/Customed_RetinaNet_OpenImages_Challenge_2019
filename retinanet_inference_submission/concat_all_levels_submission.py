# coding: utf-8
__modified_author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import pandas as pd
import numpy as np

skip_box_confidence = 0.05
iou_thr = 0.55
backbone = 'resnet50'

level1 = 'predictions_{}_{}_avg_level_1.csv'.format(skip_box_confidence, iou_thr)
level2 = 'predictions_{}_{}_avg_level_2.csv'.format(skip_box_confidence, iou_thr)
level3 = 'predictions_{}_{}_avg_level_3.csv'.format(skip_box_confidence, iou_thr)
level4 = 'predictions_{}_{}_avg_level_4.csv'.format(skip_box_confidence, iou_thr)
level5 = 'predictions_{}_{}_avg_level_5.csv'.format(skip_box_confidence, iou_thr)
saveName = 'retinanet_{}_submission_{}_{}_predictions.csv'.format(backbone, skip_box_confidence, iou_thr)

level1csv = pd.read_csv(level1)
level2csv = pd.read_csv(level2)
level3csv = pd.read_csv(level3)
level4csv = pd.read_csv(level4)
level5csv = pd.read_csv(level5)

listid = np.array(level1csv.ImageId)
predictionstring1 = level1csv.PredictionString
predictionstring2 = level2csv.PredictionString
predictionstring3 = level3csv.PredictionString
predictionstring4 = level4csv.PredictionString
predictionstring5 = level5csv.PredictionString

for idx in range(len(listid)):
    if idx % 100 == 0:
            print(idx)
    s1 = predictionstring1[idx]
    s2 = predictionstring2[idx]
    s3 = predictionstring3[idx]
    s4 = predictionstring4[idx]
    s5 = predictionstring5[idx]

    if str(s1) == 'nan':
        ensem1 = ''
    else:
        ensem1 = s1

    if str(s2) == 'nan':
        ensem2 = ''
    else:
        ensem2 = s2

    if str(s3) == 'nan':
        ensem3 = ''
    else:
        ensem3 = s3

    if str(s4) == 'nan':
        ensem4 = ''
    else:
        ensem4 = s4

    if str(s5) == 'nan':
        ensem5 = ''
    else:
        ensem5 = s5


    ensembles = ensem1 + ensem2 + ensem3 + ensem4 + ensem5
    predictionstring1[idx] = ensembles

level1csv.PredictionString = predictionstring1
level1csv.to_csv(saveName, index=False)