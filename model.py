"""SmythNet model implementation in Keras"""
from utils import *
from keras import backend as K
from keras.layers import Input, Dense, Add, Multiply, Activation, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model
import numpy as np


def build_model(shape,
                transition_matrix=None,
                accumulator_method=None,
                z_dim=None):
    timesteps, dim = shape
    y_in = Input(shape=(timesteps, dim))
    # Accumulator
    if accumulator_method is not None:
        x = Accumulator(accumulator_method, return_sequences=True)(y_in)
        wx = TimeDistributed(Dense(dim, kernel_constraint=ForceDiagonal(),
                                   use_bias=False))(x)
        latent = wx
    # Recurrent
    if z_dim is not None:
        z = LSTM(z_dim, return_sequences=True)(y_in)
        wz = TimeDistributed(Dense(dim))(z)
        latent = wz
    # Combine if Possible
    try:
        latent = Add()([wx, wz])
    except UnboundLocalError:
        pass
    # Markov
    if transition_matrix is None:
        transition_matrix = np.ones((dim, dim))
    markov = TimeDistributed(Markov(transition_matrix),
                             trainable=False)(y_in)
    # Combine Markov and Latent
    try:
        exp = TimeDistributed(Lambda(softmax_numerator, trainable=False),
                              trainable=False)(latent)
        prod = Multiply()([markov, exp])
    except UnboundLocalError:
        prod = markov
    y_out = TimeDistributed(Lambda(normalize, trainable=False),
                            trainable=False)(prod)
    model = Model(inputs=y_in, outputs=y_out)
    return model

