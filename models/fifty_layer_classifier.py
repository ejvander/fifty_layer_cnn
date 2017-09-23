from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, Conv2DTranspose, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape
from keras.layers.merge import add

from fifty_layer_base import FiftyLayerBase


class fifty_layer_classifier(FiftyLayerBase):
    def build_model(self, inp_shape=(1, 28, 28)):
        main_input = Input(shape=inp_shape, name='main_input')

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', data_format='channels_last')(main_input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

        s1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = self.build_parallel_layers(s1, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_parallel_layers(x, 128, 2)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_parallel_layers(x, 256, 2)
        #    x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        #    x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_parallel_layers(x, 512, 2)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
        x = Dense(2, activation='sigmoid')(x)
        main_output = Reshape((2,))(x)

        model = Model(input=main_input, output=main_output)

        return model
