from fifty_layer_ibis import fifty_layer_segmenter
import scripts.image_cropper as img
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model
from PIL import Image
import numpy as np
from ibis_data import ibis_data
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam
from theano import tensor as T, function, printing
from keras import backend as K
import argparse
import os
import gc
from shutil import copyfile

# Create Argument parser
parser = argparse.ArgumentParser()
# If loading the model give the name of the model to load
parser.add_argument("-l", "--load_model", help="Specify the file to load the model from")
parser.add_argument("-t", "--test_set", help="Run the testing set", action='store_true')
parser.add_argument("-r", "--train_set", help="Run the training sets", action='store_true')
parser.add_argument("-b", "--batch_num", help="Specify the batch number to start generating at", default=0)
parser.add_argument("-n", "--num_batch", help="Specify the number of batches to run", default=10)
args = parser.parse_args()

np.set_printoptions(threshold=np.nan)


def scale(x, orig_width, weight):
  """
  you taken an image and make a new image thats what? a new size
  x is image,
  why would you pass in orig_width and not just take the width from from the image?
  this fxn is also in fifty_layer_ibis_classifier_test.py
  """
  im = x.reshape(orig_width, orig_width)
  newW = int(weight*orig_width)
  im = Image.fromarray(im)
  im = im.resize((newW, newW))
  im = np.array(im)
  return im.reshape(newW, newW)

ibis_dat = ibis_data(4) # i actually dont understand what this means when you instatiate this class with the #4
 
smooth = 1 # used for calculating dice_coef
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return  -dice_coef(y_true, y_pred)

def format_imgs(X_img, Y_img, width):
    X_img = X_img.reshape(X_img.shape[0], width, width, 3) # the biggest thing slowing me down is i cant tell if this is a numpy reshape, keras reshape or some other reshape like your own. 
    Y_img = Y_img.reshape(Y_img.shape[0], width, width, 1)
    X_img = X_img.astype('float32')
    X_img /= 255
    Y_img = Y_img.astype('float32')
    Y_img /= 255

    return (X_img, Y_img)

def generate_image_sets(width=480): 
  """
  is this fxn ever called?
  """
  while 1:
    (X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)
    (X_train, Y_train) = format_imgs(X_train, Y_train, width)
    yield (X_train, Y_train)
    del X_train
    del Y_train
    X_train = None
    Y_train = None

def set_opacity(img, val):
  img=img.convert('RGBA')  #you can make sure your pic is in the right mode by check img.mode
  data=img.getdata()  #you'll get a list of tuples
  newData=[]
  for a in data:
    a=a[:3] #you'll get your tuple shorten to RGB
    if(a[0] < 127):
      a=a+(0,) # Set black bits to transparent
    else:
      a=a+(val,)
    newData.append(a)
  img.putdata(newData)
  return img

def set_color(img, val):
  img=img.convert('RGBA')  #you can make sure your pic is in the right mode by check img.mode
  data=img.getdata()  #you'll get a list of tuples
  newData=[]
  for a in data:
    alp=a[3] #you'll get your tuple shorten to RGB
    if(a[0] > 127):
      a=val+(alp,) # Replace white bits with color passed in
    newData.append(a)
  img.putdata(newData)
  return img

def prepare_comp_img(bg, ol1, ol2): # what does comp stand for? 
  ol1 = set_opacity(ol1, 100)
  ol1 = set_color(ol1, (0,255,0))
  ol2 = set_opacity(ol2, 50)
  ol2 = set_color(ol2, (0,0,255))
  comb = Image.alpha_composite(bg, ol1)
  comb = Image.alpha_composite(comb, ol2)
  return comb 

def test_img(K, X_train, Y_train, img_num, name): # this function is never called?
  test_in = X_train.reshape(1, width, width, 3)
  test_out = K.eval(model(K.variable(test_in)))
  test_out *= 255
  im1 = Image.fromarray(test_out.reshape(width, width))
  im1 = im1.convert('RGBA')
  f = open("output_images/individual/image_" + name + "_"+ str(img_num) + ".png", "w")
  im1.save(f)
  f.close()
  im2 = Image.fromarray((Y_train*255).reshape(width, width))
  im2 = im2.convert('RGBA')
  f = open("output_images/individual/image_" + name + "_" + str(img_num) + "_true.png", "w")
  im2.save(f)
  f.close()
  im3 = Image.fromarray((X_train).astype('uint8').reshape(width, width, 3))
  im3 = im3.convert('RGBA')
  f = open("output_images/individual/image_" + name + "_" + str(img_num) + "_in.png", "w")
  im3.save(f)
  f.close()

  # Save a combined image
  comb = prepare_comp_img(im3, im2, im1)
  f = open("output_images/composite/image_" + name + "_" + str(img_num) + ".png", "w")
  comb.save(f)
  f.close()

  # Save cropped image
  cropped_img = img.crop_img(im3, im1)
  f = open("output_images/classifier_images/image_" + name + "_" + str(img_num) + ".png", "w")
  cropped_img.save(f)
  f.close()

  del im1
  del im2
  del im3
  del comb
  del cropped_img

def get_testing_data(): # fxn never called?
  images = []
  for item in os.listdir("segments"):
    if "jpg" in item:
      images.append(item.split(".")[0])
  X_train = []
  Y_train = []
  image_data = np.empty(shape=(len(images), 480, 480, 3))
  seg_data = np.empty(shape=(len(images), 480, 480, 3))
  for i, im in enumerate(images):
    f = open("segments/" + im + ".jpg", "rb")
    test = Image.open(f)
    test = test.resize((480,480))
    test.load()
    f.close()
    X_train.append(np.asarray(test))
    image_data[i] = np.asarray(test) 
    del test
    test = None

    f = open("segments/" + im + "_test.png", "rb")
    test = Image.open(f)
    test = test.resize((480,480))
    test.load()
    f.close()
    test = np.asarray(test)/255
    test = test.reshape(1, 480, 480, 1)
    Y_train.append(test)
    seg_data[i] = np.asarray(test) 
    del test
    test = None
    #X_train.reshape(X_train.shape[0], width, width, 3)

  return (X_train, Y_train, image_data, seg_data)

width = 480

inp_size = (width, width, 3) # input size is width, width, 3 lol?

#fif = fifty_layer_segmenter()
#model = fif.build_model(inp_size)

#model.compile(loss=dice_coef_loss,
#              optimizer='adam',
#              metrics=['accuracy'])

#(X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)

#X_train = X_train.reshape(X_train.shape[0], width, width, 3)
#Y_train = Y_train.reshape(Y_train.shape[0], width, width, 1)
#im = Image.fromarray((X_train[0]*255).astype('uint8').reshape(width, width, 3))
#im = im.convert('RGB')
#im.save("image_test_in.png")

#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)


#for layer in model.layers:
#  print layer.name
#  print layer.input_shape
#  print layer.output_shape

#plot_model(model, to_file='model.png')
    
if(args.load_model == None):
  steps_per_ep = 100 
  nb_ep = 10
  model_out_name = 'isis_s' + str(steps_per_ep) + '-e_' + str(nb_ep) + '_1.h5' # isis lol
  
  print "Starting training. model ouput to: " + model_out_name
  fif = fifty_layer_segmenter()
  model = fif.build_model(inp_size)

  model.compile(loss=dice_coef_loss, optimizer='adam', metrics=['accuracy'])
  model.fit_generator(generate_image_sets(),
      steps_per_epoch=steps_per_ep, nb_epoch=nb_ep, max_q_size=5)
  model.save(model_out_name)
  print "Done training. model ouput to: " + model_out_name
else:
  print "Loading model: " + args.load_model
  model = load_model(args.load_model, custom_objects={'dice_coef_loss': dice_coef_loss})
  print "Done Loading model"

K.set_learning_phase(0)


# Generate results from real data
if(args.test_set):
  print "Running testing data"
  (X_test, Y_test, image_data, seg_data) = get_testing_data()
  X_image = image_data.reshape(image_data.shape[0], 480, 480, 3)
#  print "  Testing Image"
#  model.predict(X_image, batch_size=len(X_image))
  for i in xrange(0, len(X_test)):
    print "  Testing Image " + str(i), "/" + str(len(X_test))
    test_img(K, X_test[i], Y_test[i], i, "real")


# Generate results form training data
if(args.train_set):
  # Set batch to passed in argument
  ibis_dat.set_batch(int(args.batch_num))

  last = 0
  index = args.batch_num
  for b in xrange(int(args.batch_num), int(args.batch_num)+int(args.num_batch)):
    print "Running training batch #" + str(ibis_dat.batch)
    (X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)
    for i in xrange(0, len(X_train)):
      test_img(K, X_train[i], Y_train[i], i + b*ibis_dat.batch_size, "train")

    if(last):
      break


#  test_in = X_train[i].reshape(1, width, width, 3)
#  test_out = K.eval(model(K.variable(test_in)))
#  test_out *= 255
#  im1 = Image.fromarray(test_out.reshape(width, width))
#  im1 = im1.convert('RGBA')
#  im1.save("output_images/individual/image_" + str(i) + ".png")
#  im2 = Image.fromarray((Y_train[i]*255).reshape(width, width))
#  im2 = im2.convert('RGBA')
#  im2.save("output_images/individual/image_" + str(i) + "_true.png")
#  im3 = Image.fromarray((X_train[i]).astype('uint8').reshape(width, width, 3))
#  im3 = im3.convert('RGBA')
#  im3.save("output_images/individual/image_" + str(i) + "_in.png")
# 
#  comb = prepare_comp_img(im3, im2, im1)
#  comb.save("output_images/composite/image_" + str(i) + ".png")

 	
#score = model.evaluate(X_test, Y_test, verbose=1)
#print score
