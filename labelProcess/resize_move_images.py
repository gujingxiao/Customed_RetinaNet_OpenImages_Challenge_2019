import os
import shutil
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'


loadDir = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/train_3/'
saveDir = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/train/'
print(len(os.listdir(saveDir)))
count = 0
dirLength = len(os.listdir(loadDir))
for item in os.listdir(loadDir):
    name = os.path.join(loadDir, item)
    shutil.move(name, os.path.join(saveDir, item))
    count += 1
    if count % 1000 == 0:
        print(count, '/', dirLength)