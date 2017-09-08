from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import add

from custom_layers.Scale import Scale

class fifty_layer_base_w_scale:

  def build_triple_conv(self, inp, num_features, stride=1):
    x = Conv2D(num_features, (1, 1), strides=(stride, stride), padding='same', use_bias=False, data_format='channels_last')(inp)
    x = BatchNormalization(axis=1)(x)
    x = Scale(axis=1)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_features, (3, 3), padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization(axis=1)(x)
    x = Scale(axis=1)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_features*4, (1, 1), padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization(axis=1)(x)
    x = Scale(axis=1)(x)
    x = Activation('relu')(x)
    return x

  def build_parralel_layers(self, inp, num_features, stride=1):
    x1 = self.build_triple_conv(inp, num_features, stride)
    
    x2 = Conv2D(num_features*4, (1, 1), strides=(stride,stride), padding='same', use_bias=False, data_format='channels_last')(inp)
    x2 = BatchNormalization(axis=1)(x2)
    x2 = Scale(axis=1)(x2)
    x2 = Activation('relu')(x2)

    x = add([x1,x2])
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x
   
  def build_passthrough(self, inp, num_features, stride=1):
    x = self.build_triple_conv(inp, num_features, stride)
    x = add([inp, x])
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    return x

  def build_score(self, x):
    x = Conv2D(2, (1, 1), padding='same', data_format='channels_last')(x)
    return x

  def build_upscore(self, x, dim):
    x = ZeroPadding2D((1,1), data_format='channels_last')(x)
    x = Conv2DTranspose(2, (4, 4), strides=(2,2), use_bias=False, padding='full', data_format='channels_last')(x)
    return x

