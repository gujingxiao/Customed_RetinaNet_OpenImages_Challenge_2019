# coding: utf-8
__author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import os
import time
import glob
import keras
import random
import tensorflow as tf
from keras_retinanet.models.retinanet import retinanet_bbox
from labelProcess.label_levels import *
from retinanet_inference_submission.ensemble_boxes_functions import *
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet import models
from keras_retinanet.utils.visualization import draw_box, draw_caption
import pandas as pd

def show_image_debug(id_to_labels, draw, boxes, scores, labels):
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.25:
            break

        color = (0, 255, 0)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(id_to_labels[label], score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    show_image(draw)

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def get_retinanet_predictions_for_files(files, out_dir, pretrained_model_path, min_side, max_side,
                                        backbonename, label_level, show_debug_images=False):
    backbone = models.backbone(backbonename)
    print('Label Size:', len(label_level))
    model = model_with_weights(backbone.retinanet(len(label_level), modifier=None), weights=pretrained_model_path, skip_mismatch=False)
    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    print('Proc {} files...'.format(len(files)))
    count = 0
    start = time.time()
    for f in files:
        id = os.path.basename(f)[:-4]

        cache_path = out_dir + id + '.pkl'
        if os.path.isfile(cache_path):
           continue

        # preprocess image for network
        image = read_image_bgr_fast(f)

        if show_debug_images:
            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=min_side, max_side=max_side)
        image = np.expand_dims(image, axis=0)

        # process image
        boxes, scores, labels = prediction_model.predict_on_batch(image)

        if count % 100 == 0:
            print("Count: {}, Processing time: {:.2f} sec, Detections shape: {}".format(count, time.time() - start, boxes.shape))
        count += 1

        if show_debug_images:
            boxes_init = boxes.copy()
            boxes_init /= scale

        boxes[:, :, 0] /= image.shape[2]
        boxes[:, :, 2] /= image.shape[2]
        boxes[:, :, 1] /= image.shape[1]
        boxes[:, :, 3] /= image.shape[1]

        if show_debug_images:
            if len(boxes_init) > 0:
                show_image_debug(label_level, draw.astype(np.uint8), boxes_init[:1], scores[:1], labels[:1])

        save_in_file_fast((boxes, scores, labels), cache_path)


def create_csv_for_retinanet(input_dir, out_file, label_arr, skip_box_thr=0.05, intersection_thr=0.55, limit_boxes=300, type='avg'):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')
    d1, d2 = get_description_for_labels()
    files = glob.glob(input_dir + '*.pkl')
    index = 0
    for f in files:
        id = os.path.basename(f)[:-4]
        # print(id)
        out.write(id + ',')
        if os.path.exists(f):
            boxes, scores, labels = load_from_file_fast(f)
            merged_boxes = filter_boxes(boxes, scores, labels, skip_box_thr)[0]
            if len(merged_boxes) > limit_boxes:
                # sort by score
                merged_boxes = np.array(merged_boxes)
                merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

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
                #
                xmax = b[2]
                if xmax <= 0:
                    xmax = 0
                if xmax > 1:
                    xmax = 1
                #
                ymin = b[1]
                if ymin < 0:
                    ymin = 0
                if ymin > 1:
                    ymin = 1
                #
                ymax = b[3]
                if ymax <= 0:
                    ymax = 0
                if ymax > 1:
                    ymax = 1

                if (xmax <= xmin):
                    print('X min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                    continue

                if (ymax <= ymin):
                    print('Y min value larger than max value {}: {} {}'.format(label_arr[label], ymin, ymax))
                    continue

                if abs(xmax - xmin) < 1e-5:
                    print('Too small diff for {}: {} and {}'.format(label_arr[label], xmin, xmax))
                    continue

                if abs(ymax - ymin) < 1e-5:
                    print('Too small diff for {}: {} and {}'.format(label_arr[label], ymin, ymax))
                    continue

                str1 = "{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ".format(google_name, score, xmin, ymin, xmax, ymax)
                out.write(str1)
            out.write('\n')
        else:
            out.write('\n')

        if index % 1000 == 0:
            print(index, '/', len(files))
        index += 1


if __name__ == '__main__':
    gpu_use = 1
    skip_box_confidence = 0.05
    iou_thr = 0.55
    limit_boxes_per_image = 300
    show_result = False
    type = 'avg'
    backbone = 'resnet50'
    label_level = 1
    min_size = 768
    max_size = 1024
    pretrained_model_path = '../retinanet_training_oid_2019/snapshots/resnet50_oid_level_{}_06.h5'.format(label_level)
    inference_predict = True

    if label_level == 1:
        labels_list = LEVEL_1_LABELS
    elif label_level == 2:
        labels_list = LEVEL_2_LABELS
    elif label_level == 3:
        labels_list = LEVEL_3_LABELS
    elif label_level == 4:
        labels_list = LEVEL_4_LABELS
    elif label_level == 5:
        labels_list = LEVEL_5_LABELS
    else:
        raise ValueError("Label level ({}) is not supported.".format(label_level))

    output_cache_directory = ROOT_PATH + 'cache_retinanet_level_{}_{}/'.format(label_level, backbone)

    if inference_predict == True:
        print('GPU use: {}'.format(gpu_use))
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
        keras.backend.tensorflow_backend.set_session(get_session())

    files_to_process = glob.glob(ROOT_PATH + 'test/*.jpg')
    random.shuffle(files_to_process)
    if not os.path.isdir(output_cache_directory):
        os.mkdir(output_cache_directory)

    if inference_predict == True:
        print('Start making predictions. Backbone {}, level {}, Saving pkl folder: {}'.format(backbone, label_level, output_cache_directory))
        get_retinanet_predictions_for_files(files_to_process, output_cache_directory, pretrained_model_path, min_size, max_size,
                                            backbone, labels_list, show_debug_images=show_result)
        print('Finish making predictions.')
    else:
        print('Start creating level {} csv file.'.format(label_level))
        create_csv_for_retinanet(output_cache_directory, 'predictions_{}_{}_{}_level_{}444.csv'.format(skip_box_confidence, iou_thr, type, label_level),
                                 labels_list, skip_box_confidence, iou_thr, limit_boxes_per_image, type=type)