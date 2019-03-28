from models import FCN
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras import regularizers
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-cuda", "--cuda_num", type=str, default='0')
parser.add_argument("-name", "--out_name", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# The compile step specifies the training configuration.
model.compile(optimizer=keras.optimizers.SGD(lr=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', ])

# Trains for 200 epochs.
history = model.fit(x_train, y_train, validation_split=0.33, epochs=200, batch_size=256)
test = model.evaluate(x_test, y_test)

# list all data in history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig("acc_%s.png" % args.out_name)
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig("loss_%s.png" % args.out_name)
plt.close()