import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, \
     Dropout, LeakyReLU, MaxPool2D, Flatten, add, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers


class _EncoderDecoder:
    def __init__(self, activation, use_skip_conn, num_classes):
        self._activation = activation
        self._use_skip_conn = use_skip_conn
        self._num_classes = num_classes
        self._inputs = Input(shape=(32, 32, 3))

    def encoder(self, inputs):
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.2)(x)
        # Skip connection of encoder
        self._skip = Conv2D(32, 3, padding='same')(x)
        x = LeakyReLU()(self._skip)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.2)(x)

        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPool2D(name='encoded')(x)
        return encoded

    def decoder(self, encoded):
        x = Conv2DTranspose(128, 3, activation='relu', strides=(2, 2), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2DTranspose(32, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2DTranspose(32, 3, padding='same')(x)
        # adding skip connection
        if self._use_skip_conn:
            x = add([x, self._skip])
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        decoded = Conv2DTranspose(3, 3, activation=self._activation, strides=(2, 2), padding='same')(x)
        return decoded

    def classifier(self, encoded):
        x = Flatten()(encoded)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(self._num_classes, activation='softmax')(x)
        return x


    def get_model(self, return_encoder_classifier):
        # Decoder

        if return_encoder_classifier:
            model = Model(self._inputs, self.decoder(self.encoder(self._inputs)))
            classifier = Model(self._inputs, self.classifier(self.encoder(self._inputs)))
            return model, classifier

        return Model(self._inputs, self.decoder(self.encoder(self._inputs)))


def AutoEncoder(return_encoder_classifier, use_skip_conn=False, activation = 'sigmoid', num_classes=10 ):
    ae = _EncoderDecoder(activation, use_skip_conn, num_classes)
    return ae.get_model(return_encoder_classifier)


if __name__=='__main__':
    model, encoder = AutoEncoder(return_encoder_classifier=True)
    print(model.summary())