from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import add

from custom_layers.Scale import Scale

class fifty_layer:

  def build_triple_conv(self, inp, num_features, stride=1):
    x = Conv2D(num_features, (1, 1), strides=(stride, stride), padding='same', use_bias=False, data_format='channels_last')(inp)
#    x = BatchNormalization(axis=1)(x)
#    x = Scale(num_features, bias=True)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_features, (3, 3), padding='same', use_bias=False, data_format='channels_last')(x)
#    x = BatchNormalization(axis=1)(x)
#    x = Scale(num_features, bias=True)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_features*4, (1, 1), padding='same', use_bias=False, data_format='channels_last')(x)
#    x = BatchNormalization(axis=1)(x)
#    x = Scale(num_features*4, bias=True)(x)
    x = Activation('relu')(x)
    return x

  def build_parralel_layers(self, inp, num_features, stride=1):
    x1 = self.build_triple_conv(inp, num_features, stride)
    
    x2 = Conv2D(num_features*4, (1, 1), strides=(stride,stride), padding='same', use_bias=False, data_format='channels_last')(inp)
    #x2 = BatchNormalization(axis=1)(x2)
    #x2 = Scale(num_features*4, bias=True)(x2)
    x2 = Activation('relu')(x2)

    x = add([x1,x2])
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x
   
  def build_passthrough(self, inp, num_features, stride=1):
    x = self.build_triple_conv(inp, num_features, stride)
    x = add([inp, x])
#    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    return x

  def build_score(self, x):
    x = Conv2D(2, (1, 1), padding='same', data_format='channels_last')(x)
    return x

  def build_upscore(self, x, dim):
    x = ZeroPadding2D((1,1), data_format='channels_last')(x)
    x = Conv2DTranspose(2, (4, 4), strides=(2,2), use_bias=False, padding='full', data_format='channels_last')(x)
    return x

  def build_model(self, inp_shape=(1,28,28)):
    main_input = Input(shape=inp_shape, name='main_input')

    x = Conv2D(64, (7, 7), strides=(2,2), padding='same', data_format='channels_last')(main_input)
    x = BatchNormalization(axis=1)(x)
    #x = Scale(64, bias=True)(x)
    x = Activation('relu')(x)

    s1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = self.build_parralel_layers(s1, 64)
    x = self.build_passthrough(x, 64)
#    x = self.build_passthrough(x, 64)
    x = self.build_parralel_layers(x, 128, 2)
#    x = self.build_passthrough(x, 128)
#    x = self.build_passthrough(x, 128)
    o1 = self.build_passthrough(x, 128)
    x = self.build_parralel_layers(o1, 256, 2)
#    x = self.build_passthrough(x, 256)
#    x = self.build_passthrough(x, 256)
#    x = self.build_passthrough(x, 256)
    x = self.build_passthrough(x, 256)
    o2 = self.build_passthrough(x, 256)
    x = self.build_parralel_layers(o2, 512, 2)
    x = self.build_passthrough(x, 512)
#    x = self.build_passthrough(x, 512)
    x = self.build_score(x)
    x = self.build_upscore(x, 15)
    o2 = self.build_score(o2)
    x = add([x, o2])
    x = self.build_upscore(x, 30)
    o1 = self.build_score(o1)
    x = add([x, o1])

    x = ZeroPadding2D((1,1), data_format='channels_last')(x)
    main_output = Conv2DTranspose(1, (10, 10), strides=(8,8), use_bias=False, activation='sigmoid', padding='full', data_format='channels_last')(x)

    model = Model(input=main_input, output=main_output)

    return model


