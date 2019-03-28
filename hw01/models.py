import tensorflow as tf
from tensorflow.keras import layers, Model

class FCN(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(FCN, self).__init__(name='FCN')
    self.num_classes = num_classes
    # Define your layers here.
    self.flat = tf.keras.layers.Flatten()
    self.dense_1 = layers.Dense(512, activation='relu')
    self.dense_2 = layers.Dense(512, activation='relu')
    self.dense_3 = layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.flat(inputs)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return self.dense_3(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)
