import os
import pandas as pd
import numpy as np

# Level 1
# 'Man', 'Human face', 'Woman', 'Window'
# TOO_MUCH_LABELS = ['/m/04yx4', '/m/0dzct', '/m/03bt1vf', '/m/0d4v4']
# csvfile = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/output/level_1_files/train-exists-annotations-bbox-level-1.csv'
# savefile = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/output/level_1_files/train-exists-annotations-bbox-level-1-select.csv'

# Level 6
# 'Man', 'Human face', 'Woman', 'Window'
TOO_MUCH_LABELS = ['/m/07j7r', '/m/01g317', '/m/09j5n']
csvfile = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/output/level_6_files/train-exists-annotations-bbox-level-6.csv'
savefile = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/output/level_6_files/train-exists-annotations-bbox-level-6-select.csv'

boxes = pd.read_csv(csvfile)
idList = boxes['ImageID']
labelList = boxes['LabelName']

print('Analysing tooMuch Infos...')
tooMuch_boxes = boxes[boxes['LabelName'].isin(TOO_MUCH_LABELS)]
tooMuch_index_sort = tooMuch_boxes['ImageID'].value_counts().sort_index()

print('Analysing tooMuch Ori Infos...')
tooMuch_ori_boxes = boxes[boxes['ImageID'].isin(tooMuch_boxes['ImageID'].value_counts().index)]
tooMuch_ori_index_sort = tooMuch_ori_boxes['ImageID'].value_counts().sort_index()

bboxList = []
for idx in range(len(tooMuch_ori_index_sort)):
    tooMuch_name = tooMuch_index_sort.index[idx]
    tooMuch_number = tooMuch_index_sort.values[idx]
    tooMuch_ori_name = tooMuch_ori_index_sort.index[idx]
    tooMuch_ori_number = tooMuch_ori_index_sort.values[idx]

    if float(tooMuch_number) / float(tooMuch_ori_number) > 0.95:
        if idx % 10000 == 0:
            print(idx, ' / ', len(tooMuch_ori_index_sort))
        bboxList.append(tooMuch_name)

reduced_boxes = boxes[~boxes['ImageID'].isin(bboxList)]
print(boxes['LabelName'].value_counts())
print(reduced_boxes['LabelName'].value_counts())
print(len(reduced_boxes['ImageID'].value_counts().index), len(reduced_boxes['LabelName']))
reduced_boxes.to_csv(savefile, index=False)