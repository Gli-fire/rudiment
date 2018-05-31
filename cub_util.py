import os
import numpy as np
from scipy.misc import imread,imresize
import pyprind

class CUB200(object):
    """
    Util for loading the CUB-200 dataset.
    """
    def __init__(self, path, image_size=(224, 224)):
        self._path = path
        self._size = image_size

    def _classes(self):
        return os.listdir(self._path)

    def _load_image(self, category, im_name):
        return imresize(imread(os.path.join(self._path, category, im_name), mode="RGB"), self._size)

    def load_dataset(self, num_per_class=None):
        classes = self._classes()
        all_images = []
        all_labels = []
        for c in classes:
            class_images = os.listdir(os.path.join(self._path, c))

            if num_per_class is not None:
                class_images = np.random.choice(class_images, num_per_class)
            print(c)
            pbar = pyprind.ProgBar(len(class_images))
            for image_name in class_images:
                all_images.append(self._load_image(c, image_name))
                all_labels.append(c)
                pbar.update()
        #return np.array(all_images).astype(float), np.array(all_labels)
        return all_images, np.array(all_labels)
#num_per_class=4

if __name__ == "__main__":
    """
    This is just a test to make sure all is good with the world. 
    """
    import matplotlib.pyplot as plt

    CUB_DIR = 'C:\\Users\\wfr\\Desktop\\files\\AI\\PROJECT\\data'
    X, lbl = CUB200(CUB_DIR).load_dataset()
    X = np.array(X).astype(float)
    n = X.shape[0]
    rnd_birds = np.vstack([np.hstack([X[np.random.choice(n)] for i in range(10)])
                           for j in range(10)])
    plt.figure(figsize=(6, 6))
    plt.imshow(rnd_birds / 255)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title("100 random pictures...", fontsize=30)

    plt.show()



