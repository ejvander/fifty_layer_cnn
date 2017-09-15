from PIL import Image
import numpy as np
import os


class IbisData(object):
    path = "ibis/training/"
    images_path = "images/ISIC-2017_Training_Data/"
    segments_path = "segments/ISIC-2017_Training_Part1_GroundTruth/"

    segments_suffix = "_segmentation.png"

    training_size = 2000

    batch = 0

    image_list = None
    batch_size = 32

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def set_batch(self, batch):
        self.batch = batch

    def build_image_list(self):
        num_read = 0
        self.image_list = []
        for root, dirs, filenames in os.walk(self.path + self.images_path):
            for f in filenames:
                if num_read < self.training_size and "_superpixels" not in f and ".jpg" in f:
                    self.image_list.append(f)
                    num_read += 1

    def load_next_batch(self, width, height):
        if self.image_list == None:
            self.build_image_list()

        start = self.batch * self.batch_size

        end = start + self.batch_size if len(self.image_list) > start + self.batch_size \
            else len(self.image_list)

        dat_size = end - start
        image_data = np.zeros(shape=(dat_size, width, height, 3))
        segment_data = np.zeros(shape=(dat_size, width, height))
        num_read = 0
        for i in range(start, end):
            image_name = self.image_list[i]
            f = open(self.path + self.images_path + image_name)
            image = Image.open(f)
            image = image.resize((width, height))
            image.load()
            f.close()
            data = np.asarray(image)
            image_data[num_read] = data
            image.close()
            del image
            image = None

            seg_name = image_name.split(".jpg")[0] + self.segments_suffix
            f = open(self.path + self.segments_path + seg_name)
            image = Image.open(f)
            image = image.resize((width, height))
            image.load()
            f.close()
            data = np.asarray(image)
            segment_data[num_read] = data
            image.close()
            del image
            image = None

            num_read += 1

        last = 0
        if (end == len(self.image_list)):
            last = 1
            self.batch = 0
        else:
            self.batch += 1

        return (image_data, segment_data, last)

    def load_data(self, width, height):
        image_data = np.empty(shape=(self.training_size, width, height, 3))
        segment_data = np.empty(shape=(self.training_size, width, height))
        print image_data.shape
        num_read = 0
        for root, dirs, filenames in os.walk(self.path + self.images_path):
            for f in filenames:
                if num_read < self.training_size and "_superpixels" not in f and ".jpg" in f:
                    fi = open(os.path.join(root, f))
                    image = Image.open(fi)
                    image = image.resize((width, height))
                    image.load()
                    fi.close()
                    data = np.asarray(image)
                    image_data[num_read] = data
                    del image
                    image = None

                    seg_name = f.split(".jpg")[0] + self.segments_suffix
                    fi = open(os.path.join(self.path + self.segments_path, seg_name))
                    image = Image.open(fi)
                    image = image.resize((width, height))
                    fi.close()
                    data = np.asarray(image)
                    segment_data[num_read] = data
                    del image
                    image = None

                    num_read += 1
                    print "Read " + str(num_read) + " images"

        return (image_data, segment_data)
