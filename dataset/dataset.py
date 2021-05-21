from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PrepareDataset:
    def __init__(self):
        self._train_data = None
        self._train_label = None
        self._test_data = None
        self._test_label = None

        (self._train_data, self._train_label), (self._test_data, self._test_label) = cifar10.load_data()


        #self.dataset_properties()

    def dataset_properties(self):
        print('Training images shape:', self._train_data.shape)
        print('Trainining labels shape:', self._train_label.shape)
        print('Test images shape:', self._test_data.shape)
        print('Test labels shape:', self._test_label.shape)

        print(self._train_data.shape[0], 'train samples')
        print(self._test_data.shape[0], 'test samples')

        classes, counts = np.unique(self._train_label, return_counts=True)

        print("total classes ", counts, "and they are", classes)

    def filter_2500(self,
                    images,
                    labels,
                    class_num_to_filter = [2, 4, 9],
                    images_to_keep = 2500):
        # 2500 images for Bird, Deer and Truck

        if len(images) != len(labels):
            raise("Number of images and labels are not same")


        counts = {
            2 : 0,
            4 : 0,
            9 : 0
        }

        filtered_images, filtered_labels = list(), list()

        for i in tqdm(range(len(images))):
            if labels[i] in class_num_to_filter:
                if counts[int(labels[i])] < images_to_keep:
                    counts[int(labels[i])] += 1
                    filtered_images.append(images[i])
                    filtered_labels.append(labels[i])
            else:
                filtered_images.append(images[i])
                filtered_labels.append(labels[i])

        classes, counts = np.unique(filtered_labels, return_counts=True)

        print("total images per class:", counts, "for classes:", classes, "and total images", np.sum(counts))



        self._train_data = np.array(filtered_images)
        self._train_label = np.array(filtered_labels)

        return self._train_data, self._train_label


    def get_dataset(self):
        return self._train_data, self._train_label , self._test_data, self._test_label

if __name__ == '__main__':
    _ = PrepareDataset()
    #_.filter_2500()
