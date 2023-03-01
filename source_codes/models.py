"""

"""
from tensorflow import keras
from keras import layers, optimizers
from keras.applications import VGG19, vgg19, resnet, inception_resnet_v2, xception, Xception


def vgg19_siamese(img, img_size=224, unfreez_convtop_n=0):
    """

    :param name:
    :param unfreez_top_n:
    :param img:
    :param img_size:
    :return:
    """
    # Extracting features from VGG19 pretrained with 'imagenet'
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine-tuning by freezing the last 'n' convolutional layers of VGG19 (last block)
    for layer in feature_extractor.layers[:-unfreez_convtop_n]:
        layer.trainable = False

    # Convert RGB to BGR
    img = vgg19.preprocess_input(img)

    high_dim_feature = feature_extractor(img)

    return high_dim_feature


def xception_feature_extractor(img, img_size=224, unfreez_convtop_n=0):
    """

    :param unfreez_top_n:
    :param img:
    :param img_size:
    :return:
    """
    # Extracting features from VGG19 pretrained with 'imagenet'
    feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine-tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    for layer in feature_extractor.layers[:-unfreez_convtop_n]:
        layer.trainable = False

    # Convert RGB to BGR
    img = Xception.preprocess_input(img)

    high_dim_feature = feature_extractor(img)

    return high_dim_feature


