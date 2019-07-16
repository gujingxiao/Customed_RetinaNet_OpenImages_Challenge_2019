"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras.utils import get_file
import keras_resnet
import keras_resnet.models
from ..backbones import resNeXt

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class ResNeXtBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNeXtBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnext_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnext_filename = 'resnext{}_weights_tf_dim_ordering_tf_kernels_notop.h5'
        resnext_resource = 'https://github.com/keras-team/keras-applications/releases/download/resnet/'
        depth = int(self.backbone.replace('resnext', ''))

        filename = resnext_filename.format(depth)
        resource = resnext_resource

        if depth == 50:
            checksum = '62527c363bdd9ec598bed41947b379fc'
        elif depth == 101:
            checksum = '0f678c91647380debd923963594981b3'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnext50', 'resnext101']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def resnext_retinanet(num_classes, backbone='resnext50', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnext50':
        resnext = resNeXt.ResNeXt50(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'resnext101':
        resnext = resNeXt.ResNeXt101(input_tensor=inputs, include_top=False, weights=None)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnext = modifier(resnext)

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnext.outputs[0:], **kwargs)


def resnext50_retinanet(num_classes, inputs=None, **kwargs):
    return resnext_retinanet(num_classes=num_classes, backbone='resnext50', inputs=inputs, **kwargs)


def resnext101_retinanet(num_classes, inputs=None, **kwargs):
    return resnext_retinanet(num_classes=num_classes, backbone='resnext101', inputs=inputs, **kwargs)
