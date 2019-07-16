import pandas as pd

subcsv = pd.read_csv('retinanet_resnet152_101_new_ensemble_submission_0.05_0.55_predictions_all_levels.csv')

for index in range(5):
    ids = subcsv['ImageId'].values[index * 20000 : min(99999, (index + 1) * 20000)]
    preds = subcsv['PredictionString'].values[index * 20000 : min(99999, (index + 1) * 20000)]

    out = open('retinanet_resnet152_101_new_ensemble_submission_0.05_0.55_predictions_all_levels_part_{}.csv'.format(index), 'w')
    out.write('ImageId,PredictionString\n')
    for idx in range(len(ids)):
        out.write(ids[idx] + ',')
        if str(preds[idx]) == 'nan':
            out.write('')
            out.write('\n')
        else:
            out.write(preds[idx])
            out.write('\n')
    out.close()