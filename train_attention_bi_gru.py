# """
# This is the main script for the training.
# It contains speech-motion neural network implemented in Keras
# This script should be used to train the model, as described in READ.me
# """

import sys
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import keras
from keras.layers import Bidirectional



import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

# Check if script get enough parameters
if len(sys.argv) < 6:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE (DIM)')
ENCODED = sys.argv[5].lower() == 'true'

if ENCODED:
    if len(sys.argv) < 7:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE DIM')
    else:    
        N_OUTPUT = int(sys.argv[6])  # Representation dimensionality
else:
    N_OUTPUT = 192 * 2  # Number of Gesture Feature (position + velocity)


EPOCHS = int(sys.argv[2])
DATA_DIR = sys.argv[3]
N_INPUT = int(sys.argv[4])  # Number of input features

BATCH_SIZE = 2056
N_HIDDEN = 256

N_CONTEXT = 60 + 1  # The number of frames in the context

import tensorflow as tf

class MultiHeadAttentionLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

    def build(self, input_shape):
        self.q_dense = keras.layers.Dense(self.d_model)
        self.k_dense = keras.layers.Dense(self.d_model)
        self.v_dense = keras.layers.Dense(self.d_model)
        self.final_dense = keras.layers.Dense(self.d_model)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        depth = self.d_model // self.num_heads
        reshaped_x = tf.reshape(x, (batch_size, seq_len, self.num_heads, depth))
        output = tf.transpose(reshaped_x, perm=[0, 2, 1, 3])
        return output

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v)

        concat_attention = tf.reshape(scaled_attention, (batch_size, seq_len, self.d_model))
        output = self.final_dense(concat_attention)
        return output

def train(model_file):
    """
    Train a neural network to take speech as input and produce gesture as an output

    Args:
        model_file: file to store the model

    Returns:

    """

    # Get the data
    X = np.load(DATA_DIR + '/X_train.npy')

    if ENCODED:

        # If we learn speech-representation mapping we use encoded motion as output
        Y = np.load(DATA_DIR + '/' + str(N_OUTPUT)+ '/Y_train_encoded.npy')

        # Correct the sizes
        train_size = min(X.shape[0], Y.shape[0])
        X = X[:train_size]
        Y = Y[:train_size]

    else:
        Y = np.load(DATA_DIR + '/Y_train.npy')

    N_train = int(len(X)*0.9)
    N_validation = len(X) - N_train

    # Split on training and validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


    # Define Keras model
    model = Sequential()

    model.add(TimeDistributed(Dense(N_HIDDEN), input_shape=(N_CONTEXT, N_INPUT)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Add the attention layer here
    model.add(MultiHeadAttentionLayer(d_model=N_HIDDEN, num_heads=8)) # Adjust d_model and num_heads as per your need

    model.add(Bidirectional(GRU(N_HIDDEN, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    print(model.summary())

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, Y_validation))
     
    model.save(model_file)

    # Save convergence results into an image
    pyplot.plot(hist.history['loss'], linewidth=3, label='train')
    pyplot.plot(hist.history['val_loss'], linewidth=3, label='valid')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.savefig(model_file.replace('hdf5', 'png'))


if __name__ == "__main__":

    train(sys.argv[1])
