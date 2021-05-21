import numpy as np
import cv2

class Preprocess:
    def __init__(self):
        pass

    def add_noise(self, image_data):
        noise = np.random.normal(loc=0.0, scale=0.3, size=image_data.shape)
        image_data = image_data + noise * 0.3
        image_data = np.clip(image_data, 0., 1.)
        return image_data

    def normalize(self, image_data):
        return image_data / 255.0