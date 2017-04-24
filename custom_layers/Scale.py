from keras.layers import Layer
from keras import backend as K

class Scale(Layer):
  def __init__(self, nb_filter, bias, **kwargs):
    super(Scale, self).__init__(**kwargs)
    self.bias = bias
    self.nb_filter = nb_filter

  def build(self, input_shape):
    self.scale = self.add_weight(shape=(self.nb_filter,), initializer='uniform', trainable=True)
    if(self.bias):
      self.b = self.add_weight(shape=(self.nb_filter,), initializer='zero', trainable=True)
    else:
      self.b = 0;

  def call(self, x, mask=None):
    out = x*K.reshape(self.scale, (1, self.nb_filter, 1, 1))
    if(self.bias):
      out += K.reshape(self.b, (1, self.nb_filter, 1, 1))
    return out
