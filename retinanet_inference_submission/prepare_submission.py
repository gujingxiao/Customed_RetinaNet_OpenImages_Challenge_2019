import pandas as pd
import numpy as np

level1 = 'predictions_0.15_0.5_avg_level1_strength.csv'
level2 = 'predictions_0.15_0.5_avg_level2_strength.csv'
level3 = 'predictions_0.15_0.5_avg_level3_strength.csv'
level4 = 'predictions_0.15_0.5_avg_level4_strength.csv'

level1csv = pd.read_csv(level1)
level2csv = pd.read_csv(level2)
level3csv = pd.read_csv(level3)
level4csv = pd.read_csv(level4)

listid = np.array(level1csv.ImageId)
predictionstring1 = level1csv.PredictionString
predictionstring2 = level2csv.PredictionString
predictionstring3 = level3csv.PredictionString
predictionstring4 = level4csv.PredictionString

for idx in range(len(listid)):
    if idx % 100 == 0:
            print(idx)
    s1 = predictionstring1[idx]
    s2 = predictionstring2[idx]
    s3 = predictionstring3[idx]
    s4 = predictionstring4[idx]

    if type(s1) != str:
        ensem1 = ''
    else:
        ensem1 = s1 + ' '

    if type(s2) != str:
        ensem2 = ''
    else:
        ensem2 = s2 + ' '

    if type(s3) != str:
        ensem3 = ''
    else:
        ensem3 = s3 + ' '

    if type(s4) != str:
        ensem4 = ''
    else:
        ensem4 = s4


    ensembles = ensem1 + ensem2 + ensem3 + ensem4
    predictionstring1[idx] = ensembles

level1csv.PredictionString = predictionstring1
level1csv.to_csv('retinanet_resnet50_submission_0.15.csv', index=False)