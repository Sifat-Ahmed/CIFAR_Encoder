import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, \
     Dropout, LeakyReLU, MaxPooling2D, Flatten, add, Dense, UpSampling2D
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
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # 32 x 32 x 32
        x = BatchNormalization()(x)
        self._skip_conn = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(self._skip_conn)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 16 x 16 x 32
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # 16 x 16 x 64
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 8 x 8 x 64
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 8 x 8 x 128
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # 8 x 8 x 256
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        encoded = BatchNormalization()(x)
        return encoded

    def decoder(self, encoded):
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)  # 8 x 8 x 256
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 8 x 8 x 128
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # 16 x 16 x 64
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 16 x 16 x 32
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        # adding skip connection
        if self._use_skip_conn:
            x = add([x, self._skip_conn])
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)  # 32 x 32 x 32
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # 32 x 32 x 3
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