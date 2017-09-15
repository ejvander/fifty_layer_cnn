from PIL import Image
from ibis_data import IbisData
import numpy as np
import os
import random


class RandomClassifierData(IbisData):
    training_list = []
    testing_list = []

    testing_perc = 50
    num_batches = 50

    def __init__(self, batch_size):
        super(RandomClassifierData, self).__init__(batch_size)

    def build_image_list(self):
        num_read = 0
        self.image_list = []
        for root, dirs, filenames in os.walk(self.path + self.images_path):
            for f in filenames:
                if num_read < self.training_size:
                    if "train" in f:
                        self.training_list.append(f)
                    else:
                        self.testing_list.append(f)
                    num_read += 1

    def randomize_image(self, img):
        # Randomly rotate image(0,90,180,270)
        rotate_amt = random.randint(0, 3) * 90
        img = img.rotate(rotate_amt)

        # Randomly flip image
        flip_img = random.randint(0, 1)
        if flip_img == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def load_next_batch(self, width, height):
        if self.image_list is None:
            self.build_image_list()

        dat_size = self.batch_size
        image_data = np.zeros(shape=(dat_size, width, height, 3))
        classifier_data = np.zeros(shape=(dat_size, 2))
        num_read = 0
        for i in range(0, self.batch_size):
            image_name = None

            # Randomly choose training or testing
            perc = random.randint(0, 99)
            if perc < self.testing_perc:
                img_num = random.randint(0, len(self.testing_list) - 1)
                image_name = self.testing_list[img_num]
            else:
                img_num = random.randint(0, len(self.training_list) - 1)
                image_name = self.training_list[img_num]

            f = open(self.path + self.images_path + image_name)
            image = Image.open(f)
            image = image.resize((width, height))
            image.load()
            f.close()
            image = image.convert('RGB')
            image = self.randomize_image(image)
            data = np.asarray(image)
            image_data[num_read] = data
            image.close()
            del image
            image = None

            if "real" in image_name:
                classifier_data[num_read][0] = 1
            else:
                classifier_data[num_read][1] = 1

            num_read += 1

        last = 0
        if self.batch == self.num_batches:
            last = 1
            self.batch = 0
        else:
            self.batch += 1

        return image_data, classifier_data, last
