from models.fifty_layer_classifier import fifty_layer_classifier
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model
from PIL import Image
import numpy as np
from data_generators.classifier_data import classifier_data
from data_generators.random_classifier_data import random_classifier_data
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam
from theano import tensor as T, function, printing
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, auc
import argparse
import os
parser = argparse.ArgumentParser()

# If loading the model give the name of the model to load
parser.add_argument("-l", "--load_model", help="Specify the file to load the model from")

args = parser.parse_args()

np.set_printoptions(threshold=np.nan)

DEBUG = False
IMG_WIDTH = 224

def scale(x, orig_width, weight):
  im = x.reshape(orig_width, orig_width)
  newW = int(weight*orig_width)
  im = Image.fromarray(im)
  im = im.resize((newW, newW))
  im = np.array(im)
  return im.reshape(newW, newW)

classifier_dat = random_classifier_data(25)
classifier_dat.path = "output_images/classifier_images/"
classifier_dat.images_path = ""

 
smooth = 1
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return  -dice_coef(y_true, y_pred)

def format_imgs(X_img, width):
    X_img = X_img.reshape(X_img.shape[0], width, width, 3)
    X_img = X_img.astype('float32')
    X_img /= 255

    return X_img


def generate_image_sets(width=224):
  while 1:
    (X_train, Y_train, last) = classifier_dat.load_next_batch(width, width)
    X_train = format_imgs(X_train, width)
    yield (X_train, Y_train)
    del X_train
    del Y_train

def debug_model(model):
  plot_model(model, to_file='classifier.png')
  for layer in model.layers:
    print layer.name + " - " + str(layer.output_shape)


inp_size = (IMG_WIDTH, IMG_WIDTH, 3)

#for layer in model.layers:
#  print layer.name
#  print layer.input_shape
#  print layer.output_shape

#plot_model(model, to_file='model.png')
model = None
    
if(args.load_model == None):
  steps_per_ep = 50
  nb_ep = 200
  model_out_name = 'classifier_s' + str(steps_per_ep) + '-e_' + str(nb_ep) + '_1_rand.h5'
  print "Staring training. model ouput to: " + model_out_name
  fif = fifty_layer_classifier()
  model = fif.build_model(inp_size)

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  if DEBUG:
    debug_model(model)
  model.fit_generator(generate_image_sets(),
      steps_per_epoch=steps_per_ep, nb_epoch=nb_ep, max_q_size=5)
  model.save(model_out_name)
  print "Done training. model ouput to: " + model_out_name
else:
  print "Loading model: " + args.load_model
  K.set_learning_phase(0)
  model = load_model(args.load_model, custom_objects={'dice_coef_loss': dice_coef_loss})
  print "Done Loading model"

#K.set_learning_phase(0)

# Generate results from real data
#print "Running testing data"
y_pred = None
y_test = None
classifier_dat.batch_size = 5
rotate_amt = 0
flip = 0
while True:
  print "Predicting batch #%d, rotate_amt %d, flip %d" % (classifier_dat.verif_batch, rotate_amt, flip)
  (x_test, y_test_tmp, last) = classifier_dat.load_verification_batch(224, 224, rotate_amt, flip)
  x_test = format_imgs(x_test, 224)
  y_pred_tmp = model.predict(x_test, batch_size=1)
  if(y_pred is None):
    y_pred = y_pred_tmp
    y_test = y_test_tmp
  else:
    y_pred = np.append(y_pred, y_pred_tmp, axis = 0)
    y_test = np.append(y_test, y_test_tmp, axis=0)

  if(last):
    # Loop through until we have done all images in the verif set
    if(rotate_amt == 270 and flip == 1):
      break
    
    last = 0
    rotate_amt = rotate_amt+90
    flip = 1 if rotate_amt//360 == 1 or flip == 1 else 0
    rotate_amt %= 360
print y_pred[:,0]
print y_test[:,0]
fpr, tpr, _ = roc_curve(y_test[:,0], y_pred[:,0])
print fpr
print tpr
auc_val = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#for i in xrange(0, len(X_test)):
#  print "  Testing Image " + str(i), "/" + str(len(X_test))
#  test_img(K, X_test[i], Y_test[i], i, "real")

# Generate results form training data
# Set batch back to 0
classifier_dat.set_batch(0)

#last = 0
#while(not last):
#  print "Running training batch #" + str(ibis_dat.batch)
#  (X_train, Y_train, last) = ibis_dat.load_next_batch(width, width)
#  for i in xrange(0, len(X_train)):
#    test_img(K, X_train[i], Y_train[i], i, "train")

#score = model.evaluate(X_test, Y_test, verbose=1)
#print score
