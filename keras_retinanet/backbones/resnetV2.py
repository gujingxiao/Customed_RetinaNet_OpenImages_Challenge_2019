from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras_resnet
from keras_applications import get_submodules_from_kwargs
from keras_applications.imagenet_utils import _obtain_input_shape

backend = None
layers = None
models = None
keras_utils = None

def block2(x, filters, freeze_bn=True, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = keras_resnet.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, freeze=freeze_bn, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = keras_resnet.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, freeze=freeze_bn, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = keras_resnet.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, freeze=freeze_bn, name=name + '_2_bn')(x)

    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, freeze_bn=True, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, freeze_bn, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, freeze_bn, name=name + '_block' + str(i))
    x = block2(x, filters, freeze_bn, stride=stride1, name=name + '_block' + str(blocks))
    return x

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           classes=1000,
           freeze_bn=True,
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = keras_resnet.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, freeze=freeze_bn, name="conv1_bn")(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x, outputs = stack_fn(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='probs')(x)
        return x
    else:
        return models.Model(img_input, outputs, name=model_name)

def ResNet50V2(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               classes=1000,
               freeze_bn=True,
               **kwargs):
    def stack_fn(x, freeze_bn=True):
        outputs = []
        x = stack2(x, 64, 3, freeze_bn, stride1=1, name='conv2')
        outputs.append(x)
        x = stack2(x, 128, 4, freeze_bn, name='conv3')
        outputs.append(x)
        x = stack2(x, 256, 6, freeze_bn, name='conv4')
        outputs.append(x)
        x = stack2(x, 512, 3, freeze_bn, name='conv5')
        outputs.append(x)
        return x, outputs
    return ResNet(stack_fn, True, True, 'resnet50v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  classes, freeze_bn,
                  **kwargs)


def ResNet101V2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                classes=1000,
                freeze_bn=True,
                **kwargs):
    def stack_fn(x, freeze_bn=True):
        outputs = []
        x = stack2(x, 64, 3, freeze_bn, stride1=1, name='conv2')
        outputs.append(x)
        x = stack2(x, 128, 4, freeze_bn, name='conv3')
        outputs.append(x)
        x = stack2(x, 256, 23, freeze_bn, name='conv4')
        outputs.append(x)
        x = stack2(x, 512, 3, freeze_bn, name='conv5')
        outputs.append(x)
        return x, outputs
    return ResNet(stack_fn, True, True, 'resnet101v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  classes, freeze_bn,
                  **kwargs)


def ResNet152V2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                classes=1000,
                freeze_bn=True,
                **kwargs):
    def stack_fn(x, freeze_bn=True):
        outputs = []
        x = stack2(x, 64, 3, freeze_bn, stride1=1, name='conv2')
        outputs.append(x)
        x = stack2(x, 128, 8, freeze_bn, name='conv3')
        outputs.append(x)
        x = stack2(x, 256, 36, freeze_bn, name='conv4')
        outputs.append(x)
        x = stack2(x, 512, 3, freeze_bn, name='conv5')
        outputs.append(x)
        return x, outputs
    return ResNet(stack_fn, True, True, 'resnet152v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  classes, freeze_bn,
                  **kwargs)