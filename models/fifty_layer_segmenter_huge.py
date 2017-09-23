from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import add

from fifty_layer_base import FiftyLayerBase


class fifty_layer_segmenter_huge(FiftyLayerBase):
    def build_model(self, inp_shape=(1, 28, 28)):
        main_input = Input(shape=inp_shape, name='main_input')

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', data_format='channels_last')(main_input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

        s1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = self.build_parallel_layers(s1, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_passthrough(x, 64)
        x = self.build_parallel_layers(x, 128, 2)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        x = self.build_passthrough(x, 128)
        o1 = self.build_passthrough(x, 128)
        x = self.build_parallel_layers(o1, 256, 2)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        x = self.build_passthrough(x, 256)
        o2 = self.build_passthrough(x, 256)
        x = self.build_parallel_layers(o2, 512, 2)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_passthrough(x, 512)
        x = self.build_score(x)
        x = self.build_upscore(x, 15)
        o2 = self.build_score(o2)
        x = add([x, o2])
        x = self.build_upscore(x, 30)
        o1 = self.build_score(o1)
        x = add([x, o1])

        x = ZeroPadding2D((1, 1), data_format='channels_last')(x)
        main_output = Conv2DTranspose(1, (10, 10), strides=(8, 8), use_bias=False, activation='sigmoid', padding='full',
                                      data_format='channels_last')(x)

        model = Model(input=main_input, output=main_output)

        return model
