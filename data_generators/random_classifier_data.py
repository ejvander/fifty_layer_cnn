from PIL import Image
from ibis_data import IbisData
import numpy as np
import os
import random

class RandomClassifierData(IbisData):
    training_list = []
    testing_list = []
  
    verif_training_list = []
    verif_testing_list = []
    verif_list = []

    testing_perc = 50
    num_batches = 50
  
    num_training_verif = 500
    num_testing_verif = 5

    verif_batch = 0

    def __init__(self, batch_size):
        super(RandomClassifierData, self).__init__(batch_size)

    def build_image_list(self):
        num_read = 0
        self.image_list = []
        for root, dirs, filenames in os.walk(self.path + self.images_path):
            for f in filenames:
                if "train" in f:
                    if(len(self.verif_training_list) < self.num_training_verif):
                        self.verif_training_list.append(f)
                    else:
                        self.training_list.append(f)
                else:
                    if(len(self.verif_testing_list) < self.num_testing_verif):
                        self.verif_testing_list.append(f)
                    else:
                        self.testing_list.append(f)
        num_read += 1

    def randomize_image(self, img, rotate_amt = None, flip_img = None):
        # Randomly rotate image(0,90,180,270)
        if rotate_amt == None:
            rotate_amt = random.randint(0,3)*90

        img = img.rotate(rotate_amt)

        # Randomly flip image
        if flip_img == None:
            flip_img = random.randint(0,1)

        if flip_img == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img
  
    def load_verification_batch(self, width, height, rotate_amt, flip_img):

        if self.image_list == None:
            self.build_image_list()
            self.verif_list = self.verif_training_list + self.verif_testing_list
       
        num_verif_batches = ((self.num_training_verif + self.num_testing_verif)/self.batch_size)-1
          
        dat_size = self.batch_size
        image_data = np.zeros(shape=(dat_size, width, height, 3))
        classifier_data = np.zeros(shape=(dat_size, 2))
        num_read = 0
       
        for i in range(0, self.batch_size):
            image_name = None
       
            image_name = self.verif_list[self.verif_batch*self.batch_size+i]
       
            f = open(self.path + self.images_path + image_name)
            image = Image.open(f)
            image = image.resize((width, height))
            image.load()
            f.close()
            image = image.convert('RGB')
            image = self.randomize_image(image, rotate_amt, flip_img)
            data = np.asarray(image)
            image_data[num_read] = data
            image.close()
            del image
            image = None
       
            if("real" in image_name):
                classifier_data[num_read][0] = 1
            else:
                classifier_data[num_read][1] = 1
       
       
            num_read += 1 
          
        last = 0
        if(self.verif_batch == num_verif_batches):
          last = 1
          self.verif_batch = 0
        else:
          self.verif_batch += 1
     
        return image_data, classifier_data, last

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
