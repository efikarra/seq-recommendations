"""Custom Keras layers/objects"""
from keras.layers import Recurrent
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.constraints import Constraint


def softmax_numerator(x):
    return K.exp(x - K.max(x, axis=-1, keepdims=True))


def normalize(x):
    return x / K.sum(x, axis=-1, keepdims=True)


def linear(tensor):
    return tensor


def log(tensor):
    return K.log(tensor+1)


def binary(tensor):
    return K.cast(tensor > 0, dtype='float32')


def get_method(method_name):
    if method_name == 'count':
        return linear
    elif method_name == 'log-count':
        return log
    elif method_name == 'binary':
        return binary
    else:
        raise AttributeError('Invalid Accumulator Method: %s' % method_name)


class Accumulator(Recurrent):
    """Accumulates inputs

    Args:
        method: Can be 'count', 'log-count', 'binary'.
    """
    def __init__(self, method, **kwargs):
        super(Accumulator, self).__init__(**kwargs)
        self.trainable = False
        self.method = get_method(method)
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_dim = input_shape[2]
        self.units = self.input_dim
        self.input_spec[0] = InputSpec(shape=(None, None, self.input_dim))
        self.states = [None]
        self.built = True

    def preprocess_input(self, inputs, training=None):
        return inputs

    def step(self, inputs, states):
        prev_output = states[0]
        new_state = inputs + prev_output
        output = self.method(new_state)
        return output, [new_state]

    def get_config(self):
        config = {'method': self.method.__name__}
        base_config = super(Accumulator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Markov(Layer):
    """Applies matrix multiplication"""
    def __init__(self, transition_matrix, **kwargs):
        super(Markov, self).__init__(**kwargs)
        self.trainable = False
        self.transition_matrix = K.variable(transition_matrix)

    def call(self, x):
        return K.dot(x, self.transition_matrix)


class ForceDiagonal(Constraint):
    """Forces weight matrix to be diagonal"""
    def __call__(self, w):
        w *= K.eye(w.shape[0])
        return w

