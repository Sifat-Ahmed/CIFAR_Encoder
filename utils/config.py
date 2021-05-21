import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam


class Config:
    def __init__(self):

        self.use_skip_conn = False
        self.add_noise = True
        self.augmentation = False
        self.train_encoder = True

        self.epochs_a = 50
        self.epochs_c = 50

        self.learning_rate_a = 0.001
        self.learning_rate_c = 0.001

        self.optimizer_a = RMSprop(lr=self.learning_rate_a)
        self.optimizer_c = Adam(lr=self.learning_rate_c)

        # self.optimizer_a = 'rmsprop'
        # self.optimizer_c = 'adam'

        self.loss_a = 'mse'
        self.loss_c = 'categorical_crossentropy'

        self.metrics = ['accuracy']
        self.batch_size = 256
        self.shuffle = True
        self.validation_split = 0.1

        self.model_name = 'autoencoder.h5'
        self.dataset_name = 'cifar10'

        self.lr_scheduler = LearningRateScheduler(self._scheduler)
        self.checkpoint = ModelCheckpoint(self.model_name, verbose = 1, save_best_only = True)

        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    def _scheduler(self, epoch):
        if epoch < 10:
            return self.learning_rate_a
        else:
            return self.learning_rate_a * tf.math.exp(0.1 * (10 - epoch))