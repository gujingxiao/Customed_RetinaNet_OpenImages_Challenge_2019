# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from labelProcess.label_levels import *
from hashlib import md5
import glob
import os

STORAGE_PATH_TRAIN = ROOT_PATH + 'train/'
STORAGE_PATH_TEST = ROOT_PATH + 'test/'

def get_md5(fname):
    hash_md5 = md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_shape(f):
    try:
        img = pyvips.Image.new_from_file(f, access='sequential')
        shape = (img.height, img.width, img.bands)
    except:
        try:
            img = np.array(Image.open(f))
            shape = img.shape
        except:
            try:
                img = cv2.imread(f)
                shape = img.shape
            except:
                shape = (0, 0, 0)
    return shape


def get_image_stat(type):
    out_file = OUTPUT_PATH + '{}_image_params.csv'.format(type)
    out = open(out_file, 'w')
    out.write('id,width,height,channel,size,md5\n')
    if type == 'train':
        files = glob.glob(STORAGE_PATH_TRAIN + '*.jpg')
    elif type == 'test':
        files = glob.glob(STORAGE_PATH_TEST + '*.jpg')

    file_length = len(files)
    count = 0

    for f in files:
        id = os.path.basename(f)[:-4]
        if count % 1000 == 0:
            print(count, '/', file_length)
        # print('Go for {}'.format(id))
        h, w, c = get_shape(f)
        m = get_md5(f)
        sz = os.path.getsize(f)
        out.write(id)
        out.write(',' + str(w))
        out.write(',' + str(h))
        out.write(',' + str(c))
        out.write(',' + str(sz))
        out.write(',' + str(m))
        out.write('\n')
        count += 1
    out.close()


if __name__ == '__main__':
    try:
        import pyvips
    except:
        print('PYVips not available. Image parameters detection will be slow!')
    get_image_stat('test')
    get_image_stat('train')