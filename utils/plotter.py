import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Model


class Plotter:
    def __init__(self):
        pass


    def plot_sample(self,
                    sample_images,
                    rows= 3,
                    cols = 5):

        f = plt.figure(figsize=(2 * cols, 2 * rows * 2))  # defining a figure

        for i in range(rows):
            for j in range(cols):
                f.add_subplot(rows * 2, cols, (2 * i * cols) + (j + 1))  # adding sub plot to figure on each iteration
                plt.imshow(sample_images[i * cols + j])
                plt.axis("off")

        f.suptitle("Sample Data", fontsize=18)
        #plt.savefig("sample.png")

        plt.show()


    def plot_compare(self,
                     sample_images1,
                     sample_images2,
                     rows=3,
                     cols=5
                     ):

        f = plt.figure(figsize=(2 * cols, 2 * rows * 2))  # defining a figure

        for i in range(rows):
            for j in range(cols):
                f.add_subplot(rows * 2, cols, (2 * i * cols) + (j + 1))  # adding sub plot to figure on each iteration
                plt.imshow(sample_images1[i * cols + j])
                plt.axis("off")

            for j in range(cols):
                f.add_subplot(rows * 2, cols, ((2 * i + 1) * cols) + (j + 1))  # adding sub plot to figure on each iteration
                plt.imshow(sample_images2[i * cols + j])
                plt.axis("off")

        f.suptitle("Comparison", fontsize=18)
        plt.show()

    def plot_history(self, history):
        f = plt.figure(figsize=(10, 7))
        f.add_subplot()

        # Adding Subplot
        plt.plot(history.history['loss'], label="loss")  # Loss curve for training set
        plt.plot(history.history['val_loss'], label="val_loss")  # Loss curve for validation set

        plt.title("Loss Curve", fontsize=18)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.grid(alpha=0.3)
        plt.legend()
        #plt.savefig("Loss_curve_cifar10.png")
        plt.show()


    def display_activation(self, model, sample_image, col_size, row_size, act_index):
        """
           Activations: models activation map
           Col_size: Number of columns
           Row_size: Number of rowss
           Act_index: Layer Number above

           """
        layer_outs = [layer.output for layer in model.layers]
        activation_model = Model(model.input, layer_outs)

        activations = activation_model.predict(sample_image.reshape(1, 32, 32, 3))

        activation = activations[act_index]
        activation_index = 0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                activation_index += 1
        # plt.savefig('layer'+str(act_index)+'.png')
        plt.show()
