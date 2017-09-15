from models import fifty_layer_segmenter, fifty_layer_segmenter_large, fifty_layer_segmenter_huge
import scripts.image_cropper as img
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model
from PIL import Image
import numpy as np
from data_generators.ibis_data import IbisData
#from matplotlib import pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam
from keras import metrics
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
parser.add_argument("-s", "--size", help="Size of the NN(optimal, large, huge)", default="optimal")
args = parser.parse_args()

np.set_printoptions(threshold=np.nan)

IMG_WIDTH = 480
ibis_dat = IbisData(4)

def scale(x, orig_width, weight):
  im = x.reshape(orig_width, orig_width)
  newW = int(weight*orig_width)
  im = Image.fromarray(im)
  im = im.resize((newW, newW))
  im = np.array(im)
  return im.reshape(newW, newW)
 
smooth = 1
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return  -dice_coef(y_true, y_pred)

def format_imgs(X_img, Y_img, width):
    X_img = X_img.reshape(X_img.shape[0], width, width, 3)
    Y_img = Y_img.reshape(Y_img.shape[0], width, width, 1)
    X_img = X_img.astype('float32')
    X_img /= 255
    Y_img = Y_img.astype('float32')
    Y_img /= 255

    return (X_img, Y_img)

def generate_image_sets(width=480):
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

def prepare_comp_img(bg, ol1, ol2):
  ol1 = set_opacity(ol1, 100)
  ol1 = set_color(ol1, (0,255,0))
  ol2 = set_opacity(ol2, 50)
  ol2 = set_color(ol2, (0,0,255))
  comb = Image.alpha_composite(bg, ol1)
  comb = Image.alpha_composite(comb, ol2)
  return comb 

def test_img(K, X_train, Y_train, img_num, name, width):
  test_in = X_train.reshape(1, width, width, 3)
  test_out = K.eval(model(K.variable(test_in)))
  test_out *= 255
  save_images(test_out, X_train, Y_train, img_num, name, width)

def save_images(output_vals, input_vals, expected_vals, img_num, name, width):
  im1 = Image.fromarray(output_vals.reshape(width, width))
  im1 = im1.convert('RGBA')
  f = open("output_images/individual/image_" + name + "_"+ str(img_num) + ".png", "w")
  im1.save(f)
  f.close()
  im2 = Image.fromarray((expected_vals*255).reshape(width, width))
  im2 = im2.convert('RGBA')
  f = open("output_images/individual/image_" + name + "_" + str(img_num) + "_true.png", "w")
  im2.save(f)
  f.close()
  im3 = Image.fromarray((input_vals).astype('uint8').reshape(width, width, 3))
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

def get_testing_data():
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

def count_trainable_layers(model):
  num_layers = 0
  for layer in model.layers:
    # Count the number of trainable layers
    if "conv2d" in layer.name:
      num_layers += 1
#   print layer.name
#   print layer.input_shape
#   print layer.output_shape
  return num_layers


inp_size = (IMG_WIDTH, IMG_WIDTH, 3)


    
if(args.load_model == None):
  steps_per_ep = 100
  nb_ep = 100
  
  print "Staring training."
  fif = None
  if(args.size == "optimal"):
    fif = fifty_layer_segmenter.fifty_layer_segmenter()
  elif(args.size == "large"):
    fif = fifty_layer_segmenter_large.fifty_layer_segmenter_large()
  elif(args.size == "huge"):
    fif = fifty_layer_segmenter_huge.fifty_layer_segmenter_huge()

  model = fif.build_model(inp_size)

  #model.compile(loss=dice_coef_loss, optimizer='adam', metrics=['accuracy'])
  model.compile(loss="binary_crossentropy", optimizer='adam', metrics=[metrics.binary_accuracy])
  model.fit_generator(generate_image_sets(),
      steps_per_epoch=steps_per_ep, nb_epoch=nb_ep, max_q_size=5)
  num_layers = count_trainable_layers(model)
  model_out_name = 'segmenter_s' + str(steps_per_ep) + '-e_' + str(nb_ep) + '-l_' + str(num_layers) + '.h5'
  model.save("model_weights/" + model_out_name)
  print "Done training. model ouput to: model_weights/" + model_out_name
else:
  print "Loading model: " + args.load_model
  K.set_learning_phase(0)
  model = load_model(args.load_model, custom_objects={'dice_coef_loss': dice_coef_loss})
  print model.summary()
  print "Done Loading model"

num_layers = count_trainable_layers(model)
print "Total Trainable layers: " + str(num_layers)

plot_model(model, to_file='segmenter.png')
K.set_learning_phase(0)


# Generate results from real data
if(args.test_set):
  print "Running testing data"
  (X_test, Y_test, image_data, seg_data) = get_testing_data()
  X_image = image_data.reshape(image_data.shape[0], 480, 480, 3)
#  print "  Testing Image"
  y_pred = model.predict(X_image, batch_size=len(X_test))
  for i in xrange(0, len(X_test)):
    save_images(y_pred[i]*255, X_test[i], Y_test[i], i, "real", IMG_WIDTH)
  #for i in xrange(0, len(X_test)):
  #  print "  Testing Image " + str(i), "/" + str(len(X_test))
  #  test_img(K, X_test[i], Y_test[i], i, "real")


# Generate results form training data
if(args.train_set):
  # Set batch to passed in argument
  ibis_dat.set_batch(int(args.batch_num))

  last = 0
  index = args.batch_num
  for b in xrange(int(args.batch_num), int(args.batch_num)+int(args.num_batch)):
    print "Running training batch #" + str(ibis_dat.batch)
    (X_train, Y_train, last) = ibis_dat.load_next_batch(IMG_WIDTH, IMG_WIDTH)
    for i in xrange(0, len(X_train)):
      test_img(K, X_train[i], Y_train[i], i + b*ibis_dat.batch_size, "train", IMG_WIDTH)

    if(last):
      break

