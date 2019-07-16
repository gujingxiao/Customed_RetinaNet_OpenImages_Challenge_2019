import os
import pandas as pd
import random
from labelProcess.label_levels import ROOT_PATH

for level in range(1, 8):
    csvfile = ROOT_PATH + 'output/level_{}_files/train-annotations-bbox-level-{}.csv'.format(level, level)
    imagePath = ROOT_PATH + 'train/'

    boxes = pd.read_csv(csvfile)
    print('**************************************************')
    print('Level {} has total {} boxes.'.format(level, len(boxes)))

    imageList = os.listdir(imagePath)
    imageNameList = []
    for item in imageList:
        itemName = item.replace('.jpg', '')
        imageNameList.append(itemName)

    reduced_boxes = boxes[boxes['ImageID'].isin(imageNameList)]
    print('Image folder has Level {} {} Images, {} boxes.'.format(level, len(reduced_boxes['ImageID'].value_counts()),len(reduced_boxes)))

    # Get Validation ID
    if level == 1:
        ratio = 0.007
    elif level == 2:
        ratio = 0.015
    elif level == 3:
        ratio = 0.03
    elif level == 4:
        ratio = 0.04
    elif level == 5:
        ratio = 0.03
    elif level == 6:
        ratio = 0.006
    elif level == 7:
        ratio = 0.03

    savefile_val = ROOT_PATH + 'output/level_{}_files/val-exists-annotations-bbox-level-{}.csv'.format(level, level)
    savefile_train = ROOT_PATH + 'output/level_{}_files/train-exists-annotations-bbox-level-{}.csv'.format(level, level)

    boxes = reduced_boxes
    imageList = os.listdir(imagePath)
    random.shuffle(imageList)

    valImageList = imageList[0: int(ratio * len(imageList))]
    trainImageList = imageList[int(ratio * len(imageList)):]

    #Val
    imageNameList = []
    for item in valImageList:
        itemName = item.replace('.jpg', '')
        imageNameList.append(itemName)
    reduced_boxes = boxes[boxes['ImageID'].isin(imageNameList)]
    print('Level {} Val has {} images, total {} boxes.'.format(level, len(reduced_boxes['ImageID'].value_counts()), len(reduced_boxes)))
    reduced_boxes.to_csv(savefile_val, index=False)

    #Train
    imageNameList = []
    for item in trainImageList:
        itemName = item.replace('.jpg', '')
        imageNameList.append(itemName)

    reduced_boxes = boxes[boxes['ImageID'].isin(imageNameList)]
    print('Level {} Train has {} images, total {} boxes.'.format(level, len(reduced_boxes['ImageID'].value_counts()), len(reduced_boxes)))
    reduced_boxes.to_csv(savefile_train, index=False)
