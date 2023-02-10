import tensorflow as tf
from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False, use_cbam=False, cbam_ratio=8):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)
    cbam_name = 'decoder_stage{}_cbam'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        if use_cbam:
            x = CBAMBlock(filters, ratio=cbam_ratio, name=cbam_name)(x)
    
        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False, use_cbam=False, cbam_ratio=8):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)
    cbam_name = 'decoder_stage{}_cbam'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
        if use_cbam:
            x = CBAMBlock(filers, ratio=cbam_ratio, name=cbam_name)(x)

        return x

    return layer

def CBAMBlock(filters,ratio=8, name='cbam'):
    layers=tf.keras.layers
    image_data_format=tf.keras.backend.image_data_format()
    if not hasattr(CBAMBlock,'counter'): CBAMBlock.counter=0
    CBAMBlock.counter+=1
    if image_data_format=='channels_first':
        axis = 1
        sig_reshape=(-1,filters,1,1)
    else:
        axis = -1
        sig_reshape=(-1,1,1,filters)
    conv2d = layers.Conv2D(filters=1,kernel_size=7, padding="same",
                                    activation='sigmoid',name=f'{name}_conv2d_{CBAMBlock.counter}')
    concat = layers.Concatenate(axis=axis, name=f'{name}_concatenate_{CBAMBlock.counter}')
    mult1 = layers.Multiply(name=f'{name}_multiply1_{CBAMBlock.counter}')
    mult2 = layers.Multiply(name=f'{name}_multiply2_{CBAMBlock.counter}')
    gap2d = layers.GlobalAveragePooling2D(data_format=image_data_format, name=f'{name}_gap2d_{CBAMBlock.counter}')
    l1 = layers.Dense(filters//ratio, activation='relu', use_bias=False, name=f'{name}_dense1_{CBAMBlock.counter}')
    l2 = layers.Dense(filters, use_bias=False, name=f'{name}_dense2_{CBAMBlock.counter}')
    add = layers.Add(name=f'{name}_add_{CBAMBlock.counter}')
    activation = layers.Activation('sigmoid',name=f'{name}_sigmoid_{CBAMBlock.counter}')

    def layer(input):
        x1 = l2(l1(gap2d(input)))
        x2 = l2(l1(gap2d(input)))
        features = mult1([input,tf.reshape(activation(add([x1,x2])),sig_reshape)])


        x3 = tf.expand_dims(tf.reduce_mean(features,axis=axis),axis=axis)
        x4 = tf.expand_dims(tf.reduce_max(features,axis=axis),axis=axis)
        x = mult2([input,conv2d(concat([x3,x4]))])
        return x
    return layer

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        use_cbam=False,
        cbam_ratio=8
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm,
                          use_cbam=use_cbam, cbam_ratio=cbam_ratio)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        decoder_use_cbam=True,
        decoder_cbam_ratio=8,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        decoder_use_cbam: if ``True``, ``CBAM`` layer after the ``Activation`` layer is used.
        decoder_cbam_ratio: if ``True``, the ``CBAM`` channel ratio is set. Ignored if decoder_use_cbam is ``False``.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """


    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        use_cbam=decoder_use_cbam,
        cbam_ratio=decoder_cbam_ratio
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
