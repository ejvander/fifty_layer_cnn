from fifty_layer_ibis import fifty_layer
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
parser = argparse.ArgumentParser()
# If loading the model give the name of the model to load
parser.add_argument("-l", "--load_model", help="Specify the file to load the model from")
args = parser.parse_args()

import memory_profiler
np.set_printoptions(threshold=np.nan)

def scale(x, orig_width, weight):
  im = x.reshape(orig_width, orig_width)
  newW = int(weight*orig_width)
  im = Image.fromarray(im)
  im = im.resize((newW, newW))
  im = np.array(im)
  return im.reshape(newW, newW)

ibis_dat = ibis_data(4)
 
smooth = 1
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return  -dice_coef(y_true, y_pred)

def generate_image_sets(width=480):
  while 1:
    (X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)
    X_train = X_train.reshape(X_train.shape[0], width, width, 3)
    Y_train = Y_train.reshape(Y_train.shape[0], width, width, 1)
    X_train = X_train.astype('float32')
    X_train /= 255
    Y_train = Y_train.astype('float32')
    Y_train /= 255
    #print "returning data"
    #print X_train.shape
    #print Y_train.shape
    yield (X_train, Y_train)
    del X_train
    del Y_train
    #print "Getting next data"
    #test_in = X_train[0].reshape(1, 3, width, width)
    #test_out = K.eval(model(K.variable(test_in)))
    #print test_out

width = 480

inp_size = (width, width, 3)

fif = fifty_layer()
model = fif.build_model(inp_size)

model.compile(loss=dice_coef_loss,
              optimizer='adam',
              metrics=['accuracy'])

(X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)
#(X_train, Y_train) = ibis_data.load_data(width, width)
#fig = plt.figure()
#a = fig.add_subplot(1,2,1)
#plt.imshow(X_train[0])
#a = fig.add_subplot(1,2,2)
#plt.imshow(Y_train[0])
#plt.show()

#print X_train.shape
#print Y_train.shape
X_train = X_train.reshape(X_train.shape[0], width, width, 3)
Y_train = Y_train.reshape(Y_train.shape[0], width, width, 1)
im = Image.fromarray((X_train[0]*255).astype('uint8').reshape(width, width, 3))
im = im.convert('RGB')
im.save("image_test_in.png")
#X_test = X_test.reshape(X_test.shape[0], 3, width, width)
#print Y_train.shape


#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255

#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)


#for layer in model.layers:
#  print layer.name
#  print layer.input_shape
#  print layer.output_shape

#plot_model(model, to_file='model.png')
    
#print test_out
#from matplotlib import pyplot as plt
#plt.imshow(test_out.reshape(width,width))
#plt.show()

if(args.load_model == None):
  model.fit_generator(generate_image_sets(),
      steps_per_epoch=50, nb_epoch=30, max_q_size=5)
  model.save('isis_8-e_400.h5')
else:
  model = load_model(args.load_model)
#model.fit(X_train, Y_train, 
#          batch_size=32, nb_epoch=5, verbose=1)
gen_obj = generate_image_sets()
(X_train, Y_train) = next(gen_obj)
  
K.set_learning_phase(0)

for i in xrange(0, len(X_train)):
  test_in = X_train[i].reshape(1, width, width, 3)
  test_out = K.eval(model(K.variable(test_in)))
  test_out *= 255
  im = Image.fromarray(test_out.reshape(width, width))
  im = im.convert('RGB')
  im.save("image_" + str(i) + ".png")
  im = Image.fromarray((Y_train[i]*255).reshape(width, width))
  im = im.convert('RGB')
  im.save("image_" + str(i) + "_true.png")
  im = Image.fromarray((X_train[i]*255).astype('uint8').reshape(width, width, 3))
  #im = im.convert('RGB')
  im.save("image_" + str(i) + "_in.png")
  fig = plt.figure()
  a = fig.add_subplot(1,3,1)
  plt.imshow(X_train[i])
  a = fig.add_subplot(1,3,2)
  plt.imshow(Y_train[i].reshape(width, width))
  a = fig.add_subplot(1,3,3)
  #print test_out
  plt.imshow(test_out.reshape(width, width))
  plt.show()
 	
#score = model.evaluate(X_test, Y_test, verbose=1)
#print score
