from dataset.dataset import PrepareDataset
from models.autoencoder2 import AutoEncoder
from preprocessing.preprocess import Preprocess
from utils.config import Config
from utils.plotter import Plotter
from tensorflow.keras.models import load_model, save_model, Model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from utils.utils import copy_weights, classification_stats
import random


def train_encoder(cfg, model, x_train, y_train, x_val, y_val):

    model.compile(optimizer=cfg.optimizer_a,
                  loss=cfg.loss_a,
                  metrics=cfg.metrics)


    #model.summary()

    if cfg.augmentation:
        cfg.datagen.fit(x_train)

        history = model.fit(cfg.datagen.flow(x_train, y_train, batch_size=cfg.batch_size),
                            validation_data=(x_val, y_val),
                            steps_per_epoch=len(x_train) / cfg.batch_size,
                            epochs=cfg.epochs_a,
                            shuffle=cfg.shuffle,
                            callbacks=[cfg.checkpoint, cfg.lr_scheduler])

    else:
        history = model.fit(x_train,
                            y_train,
                            validation_data = (x_val, y_val),
                            epochs=cfg.epochs_a,
                            batch_size=cfg.batch_size,
                            shuffle=cfg.shuffle,
                            callbacks=[cfg.checkpoint, cfg.lr_scheduler])

    return history

def train_classifier(cfg, model, x_train, y_train):
    model.compile(optimizer=cfg.optimizer_c,
                  loss=cfg.loss_c,
                  metrics=cfg.metrics)

    history = model.fit(x_train,
                        y_train,
                        validation_split=0.1,
                        epochs=cfg.epochs_c,
                        batch_size=cfg.batch_size,
                        shuffle=cfg.shuffle,
                        )

    return history

def test_encoder(x_test, y_test, model, plotter):
    y_pred = model.predict(x_test)
    error = list()
    for i in range(len(x_test)):
        #mse = np.mean(np.power(x_test[i] - y_pred[i], 2), axis=-1)
        #err = np.linalg.norm(x_test[i] - y_pred[i])
        err = np.mean(np.sum((x_test[i].astype("float") - y_pred[i].astype("float")) ** 2))

        error.append(err)

    #print(np.array(error).shape, y_test.shape)

    #class label
    y_test = [i[0] for i in y_test.tolist()]

    test_error = pd.DataFrame.from_dict({'error': error,'class': y_test})
    test_error = test_error.reset_index()

    print(test_error.describe())

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(test_error['class'], test_error['error'], c=pd.Categorical(test_error['class']).codes, cmap='tab20b')
    # plt.show()

    #sns.catplot(data=test_error, x='class', y='error', height=10, aspect=1.5)



    threshold = 5.0


    error_samples = list()
    original_samples = list()

    for i in range(len(test_error)):
        if test_error.loc[i, 'error'] > threshold:
            error_samples.append(y_pred[i])
            original_samples.append(x_test[i])

    #plotter.plot_compare(original_samples, error_samples)


def test_classifier(cfg, model, x_test, y_test, plotter):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(np.round(y_pred),axis=1).tolist()


    classification_stats(y_pred, y_test)
    plotter.display_activation(model, x_test[random.randint(0, len(x_test))].reshape(1, 32,32,3), 5, 5, 2)



def main():

    dataset = PrepareDataset()
    cfg = Config()
    plotter = Plotter()
    preprocessor = Preprocess()




    train_img, train_label, test_img, test_label = dataset.get_dataset()

    train_img, train_label = dataset.filter_2500(images = train_img,
                                           labels = train_label,
                                           class_num_to_filter = [2, 4, 9],
                                           images_to_keep = 2500)

    # dataset.plot_sample(x_train)


    x_train = preprocessor.normalize(train_img)
    y_train = x_train.copy()

    if cfg.add_noise:
        x_train = preprocessor.add_noise(x_train)
    # dataset.plot_sample(x_train)
    x_test = preprocessor.normalize(test_img)
    x_val = x_test.copy()
    y_val = x_test.copy()

    model, classifier = AutoEncoder(return_encoder_classifier=True, use_skip_conn=cfg.use_skip_conn)


    #model = EncoderWithSkipConn()


    if os.path.exists(cfg.model_name):
        print("Saved Model found, Loading...")
        model = load_model(cfg.model_name)

    if not os.path.exists(cfg.model_name) and cfg.train_encoder == False:
        raise('No saved model')


    if cfg.train_encoder:
        history = train_encoder(cfg,
                        model,
                        x_train,
                        y_train,
                        x_val,
                        y_val)

        # plotter.plot_history(history)
    test_encoder(x_test, test_label, model, plotter)


    train_img = preprocessor.normalize(train_img)
    train_label = to_categorical(train_label)


    #print(train_label)

    classifier = copy_weights(model, classifier)


    history = train_classifier(cfg,
                               classifier,
                               train_img,
                               train_label)

    plotter.plot_history(history)

    test_img = preprocessor.normalize(test_img)
    test_label = [i[0] for i in test_label.tolist()]
    test_classifier(cfg, classifier, test_img, test_label, plotter)

if __name__ == "__main__":

    # import tensorflow as tf
    #
    # if tf.test.gpu_device_name():
    #     print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    #
    # else:
    #     print("Please install GPU version of TF")


    main()