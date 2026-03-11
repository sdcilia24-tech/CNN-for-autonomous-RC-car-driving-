import tensorflow as tf
class Model(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.layer_zero = tf.keras.Sequential([tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3,3),
        activation = 'relu',
        padding = 'same',
        input_shape = (96, 96, 1),
        strides = 2
    ), tf.keras.layers.BatchNormalization()])

    self.layer_one = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
            filters = 32,
            kernel_size = (3,3),
            padding = 'same',
            activation = 'relu',
            strides = 1
        ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_two = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
            filters = 32,
            kernel_size = (3,3),
            padding = 'same',
            activation = 'relu',
            strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_three = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
            filters = 64,
            kernel_size = (3,3),
            padding = 'same',
            activation = 'relu',
            strides = 2
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_four = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 64,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_five = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 64,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
                                           ])
    self.layer_six = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 64,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_seven = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 128,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 2
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_eight = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 128,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_nine = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 128,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_ten = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 128,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_eleven = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 128,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_twelve = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 256,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 1
    ), tf.keras.layers.BatchNormalization()
    ])
    self.layer_thirteen = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(
          filters = 256,
          kernel_size = (3,3),
          padding = 'same',
          activation = 'relu',
          strides = 2
    ), tf.keras.layers.BatchNormalization()
    ])
    self.gap = tf.keras.layers.GlobalAveragePooling2D()
    self.dense = tf.keras.layers.Dense(
        units = 1,
        activation = 'tanh'
    )
  def call(self, x):

      x = self.layer_zero(x)
      x = self.layer_one(x)
      x = self.layer_two(x)
      x = self.layer_three(x)
      x = self.layer_four(x)
      x = self.layer_five(x)
      x = self.layer_six(x)
      x = self.layer_seven(x)
      x = self.layer_eight(x)
      x = self.layer_nine(x)
      x = self.layer_ten(x)
      x = self.layer_eleven(x)
      x = self.layer_twelve(x)
      x = self.layer_thirteen(x)

      x = self.gap(x)
      x = self.dense(x)

      return x
                                                  
