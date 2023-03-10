import keras_tuner as kt
import keras
from keras import layers, optimizers
from keras.applications import VGG19, vgg19, Xception, xception
from keras.regularizers import l1_l2


class SiameseHyperModelVgg19(kt.HyperModel):
    """

    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        """

        :param hp:
        :return: hyperparameter-tuning model
        """
        # data augmentation
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ]
        )

        global x

        img_size = 224

        # Definition of the 2 inputs
        img_a = layers.Input(shape=(img_size, img_size, 3), name="left_image")
        img_b = layers.Input(shape=(img_size, img_size, 3), name="right_image")

        # Convert RGB to BGR
        img_a = vgg19.preprocess_input(img_a)
        img_b = vgg19.preprocess_input(img_b)

        # Data augmentation
        img_a = data_augmentation(img_a)
        img_b = data_augmentation(img_b)

        # Feature extraction; note unfreezing conv top is at risk of overfitting
        #   Extracting features from VGG19 pretrained with 'imagenet'
        feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

        #   Fine-tuning by freezing the last 'n' convolutional layers of VGG19
        unfreez_convtop_n = hp.Int(name="num_ft_vgg19", min_value=0, max_value=8, step=1)
        for layer in feature_extractor.layers[:-unfreez_convtop_n]:
            layer.trainable = False

        out_a = feature_extractor(img_a)
        out_b = feature_extractor(img_b)

        # Concatenation of the inputs
        x = layers.concatenate([out_a, out_b])

        # Top identifier tuning
        #    convblock(convlayer + dropout)
        num_convblock = hp.Int(name="num_convblock", min_value=0, max_value=3, step=1)
        for i in range(num_convblock):
            x = layers.Conv2D(
                filters=hp.Int(name=f'filters_convblock_{i}', min_value=128, max_value=512, step=128),
                kernel_size=hp.Choice(f'kernel_size_convblock_{i}', [3, 5]),
                kernel_regularizer=l1_l2(hp.Choice(f'conv_{i}_reg_strength', values=[0.01, 0.1]))
            )(x)
            x = layers.Dropout(hp.Float(f'dropout_cb_{i}', min_value=0.0, max_value=0.5, step=0.1))(x)

        x = layers.Flatten()(x)

        #   densely connected layers
        num_dc_layer = hp.Int(name="num_convblock", min_value=0, max_value=2, step=1)
        for i in range(num_dc_layer):
            x = layers.Dense(units=hp.Int(name=f'units_dc_layer_{i}', min_value=128, max_value=512, step=64))(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="Final_dense")(x)

        # Optimizer tuning
        optimizer = optimizers.SGD(
            learning_rate=hp.Choice('learning_rate', [1e-4, 1e-5, 1e-6]),
            decay=hp.Choice('learning_rate', [1e-5, 1e-6, 1e-7]),
            momentum=hp.Choice('learning_rate', [0.68, 0.69, 0.70]),
            nesterov=True
        )

        hp_model = keras.Model(inputs=[img_a, img_b], outputs=outputs)

        hp_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        return hp_model


class SiameseHyperModelXception(kt.HyperModel):

    def __int__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        """

        :param hp:
        :return: hyperparameter-tuning model
        """
        # data augmentation
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ]
        )

        global x

        img_size = 224

        # Definition of the 2 inputs
        img_a = layers.Input(shape=(img_size, img_size, 3), name="left_image")
        img_b = layers.Input(shape=(img_size, img_size, 3), name="right_image")

        # Convert RGB to BGR
        img_a = vgg19.preprocess_input(img_a)
        img_b = vgg19.preprocess_input(img_b)

        # Data augmentation
        img_a = data_augmentation(img_a)
        img_b = data_augmentation(img_b)

        # Feature extraction; note unfreezing conv top is at risk of overfitting
        #   Extracting features from VGG19 pretrained with 'imagenet'
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

        #   Fine-tuning by freezing the last 'n' convolutional layers of VGG19
        unfreez_convtop_n = hp.Int(name="num_ft_xception", min_value=0, max_value=8, step=1)
        for layer in feature_extractor.layers[:-unfreez_convtop_n]:
            layer.trainable = False

        # Extracting features from VGG19 pretrained with 'imagenet'
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

        # Fine-tuning by freezing the last 4 convolutional layers of VGG19 (last block)
        for layer in feature_extractor.layers[:-unfreez_convtop_n]:
            layer.trainable = False

        # xception preprocessing
        img_a = xception.preprocess_input(img_a)
        img_b = xception.preprocess_input(img_b)

        # feature extractor
        out_a = feature_extractor(img_a)
        out_b = feature_extractor(img_b)

        # Concatenation of the inputs
        x = layers.concatenate([out_a, out_b])

        # Top identifier tuning
        #    convblock(convlayer + dropout)
        num_convblock = hp.Int(name="num_convblock", min_value=0, max_value=3, step=1)
        for i in range(num_convblock):
            x = layers.Conv2D(
                filters=hp.Int(name=f'filters_convblock_{i}', min_value=128, max_value=512, step=128),
                kernel_size=hp.Choice(f'kernel_size_convblock_{i}', [3, 5]),
                kernel_regularizer=l1_l2(hp.Choice(f'conv_{i}_reg_strength', values=[0.01, 0.1]))
            )(x)
            x = layers.Dropout(hp.Float(f'dropout_cb_{i}', min_value=0.0, max_value=0.5, step=0.1))(x)

        x = layers.Flatten()(x)

        #   densely connected layers
        num_dc_layer = hp.Int(name="num_convblock", min_value=0, max_value=2, step=1)
        for i in range(num_dc_layer):
            x = layers.Dense(units=hp.Int(name=f'units_dc_layer_{i}', min_value=128, max_value=512, step=64))(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="Final_dense")(x)

        # Optimizer tuning
        optimizer = optimizers.SGD(
            learning_rate=hp.Choice('learning_rate', [1e-4, 1e-5, 1e-6]),
            decay=hp.Choice('learning_rate', [1e-5, 1e-6, 1e-7]),
            momentum=hp.Choice('learning_rate', [0.68, 0.69, 0.70]),
            nesterov=True
        )

        hp_model = keras.Model(inputs=[img_a, img_b], outputs=outputs)

        hp_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        return hp_model
