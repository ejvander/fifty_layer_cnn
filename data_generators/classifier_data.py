from PIL import Image
from ibis_data import ibis_data
import numpy as np
import os

class classifier_data(ibis_data):

  def __init__(self, batch_size):
    super(classifier_data, self).__init__(batch_size)
  
  def build_image_list(self):
    num_read = 0
    self.image_list = []
    for root, dirs, filenames in os.walk(self.path + self.images_path):
      for f in filenames:
        if num_read < self.training_size:
          self.image_list.append(f)
          num_read += 1 

  def load_next_batch(self, width, height):
    if self.image_list == None:
      self.build_image_list()

    start = self.batch*self.batch_size

    end = start + self.batch_size if len(self.image_list) > start+self.batch_size \
                                  else len(self.image_list)

    dat_size = end-start
    image_data = np.zeros(shape=(dat_size, width, height, 3))
    classifier_data = np.zeros(shape=(dat_size, 2))
    num_read = 0
    for i in range(start, end):
      image_name = self.image_list[i]
      f = open(self.path + self.images_path + image_name)
      image = Image.open(f)
      image = image.resize((width, height))
      image.load()
      f.close()
      image = image.convert('RGB') 
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
    if(end == len(self.image_list)):
      last = 1
      self.batch = 0
    else:
      self.batch += 1

    return (image_data, classifier_data, last)

