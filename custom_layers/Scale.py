from keras.engine import Layer, InputSpec
from keras import backend as K


class Scale(Layer):
    uid_ = 1
    '''Learns a set of weights and biases used for scaling the input data.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)
        self.name = "Scale_" + str(Scale.uid_)
        Scale.uid_ += 1

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = self.add_weight(shape=shape, initializer=self.gamma_init, trainable=True,
                                     name="{}_gamma".format(self.name))
        self.beta = self.add_weight(shape=shape, initializer=self.beta_init, trainable=True,
                                    name="{}_beta".format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
