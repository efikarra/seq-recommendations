"""
@author: efi
"""
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Activation
from keras.layers.core import Dense, Masking
from keras.layers.merge import concatenate, add
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model


class BaseModel():
    def __init__(self, n_classes, model_name="test_model", rnn_type='simpleRNN'):
        self.model_name = model_name
        self.rnn_type = rnn_type
        self.n_classes = n_classes
        self.model = None

    def fit_model(self, x_train, y_train, validation_data=None, loss='categorical_crossentropy', metrics=[], optimizer='adam', n_epochs=10, batch_size=100, plot_history=False,
                  verbose=1):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[loss] + metrics)
        history = self.model.fit(x_train, y_train, validation_data=validation_data, epochs=n_epochs,
                                 batch_size=batch_size, verbose=verbose)
        # print fileprefix+": "+self.evaluate(X_test,Y_test)
        for key, value in history.history.items():
            print key, value
        if plot_history:
            self.plot_history(history)
        return history

    def predict(self, x_test, batch_size=10, verbose=1):
        return self.model.predict(x_test, batch_size=10, verbose=verbose)

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test)
        return (self.model.metrics_names, scores)

    def plot_history(self, history):
        plt.plot(history.history['loss'], color='b', label='Train loss')
        plt.plot(history.history['val_loss'], color='g', label='Val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig(self.model_name + 'loss')
        plt.close()

    def save_model_weights(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = h5py.File(directory + self.model_name + ".h5", 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight' + str(i), data=weight[i])
        f.close()
        print("Saved model weights to disk: " + directory + self.model_name)

    def load_model_weights(self, filepath):
        # workaround to load weights into new model
        f = h5py.File(filepath, 'r')
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight' + str(i)][:])
        self.model.set_weights(weight)

    def get_layer_weights(self, name):
        return self.model.get_layer(name).get_weights()

    def set_layer_weights_trainable(self, name, trainable=True):
        self.model.get_layer(name).trainable = trainable

    def set_layer_weights(self, name, weights):
        self.model.get_layer(name).set_weights(weights)

    def get_model_weights(self):
        return self.model.trainable_weights,self.model.non_trainable_weights

class RNNBaseline(BaseModel):
    def __init__(self, timesteps, features, n_classes, model_name="baseline_model", rnn_type='simpleRNN', out_activation="softmax", z_activation="relu", z_dim=20):
        BaseModel.__init__(self, n_classes, model_name=model_name, rnn_type=rnn_type)
        input = Input(shape=(timesteps, features), name='input')
        masked_input = Masking(mask_value=0.0, name="mask")(input)
        if self.rnn_type == 'simpleRNN':
            rnn_model = SimpleRNN(z_dim, input_shape=(timesteps, features), return_sequences=True,
                                  activation=z_activation,
                                  name="rnn")

        if self.rnn_type == 'LSTM':
            rnn_model = LSTM(z_dim, input_shape=(timesteps, features), return_sequences=True, activation=z_activation,
                             name='lstm')
        rnn_output = rnn_model(masked_input)
        output = TimeDistributed(Dense(n_classes, activation=out_activation), name="output")(masked_input)
        self.model = Model(inputs=input, outputs=output)


class RNNY2YModel(BaseModel):
    def __init__(self, timesteps, x_dim, y_dim, z_dim=20, model_name="y_to_y_model", rnn_type='simpleRNN', z_to_z_activation="relu",
                 y_to_y_activation="linear", xz_to_y_activation="linear", out_activation="softmax", optimizer='adam',
                 y_bias=False, z_bias=True, xz_bias=False):
        BaseModel.__init__(self, y_dim, model_name=model_name, rnn_type=rnn_type)

        y_input = Input(shape=(timesteps, y_dim), name="y_input")
        # build rnn model
        mask = Masking(mask_value=0.0, input_shape=(timesteps, y_dim), name="mask")
        masked_y_input = mask(y_input)
        if self.rnn_type == 'simpleRNN':
            rnn_model = SimpleRNN(z_dim, input_shape=(timesteps, x_dim), return_sequences=True,
                                  activation=z_to_z_activation,
                                  name="x_to_z_rnn")

        if self.rnn_type == 'LSTM':
            rnn_model = LSTM(z_dim, input_shape=(timesteps, x_dim), return_sequences=True, activation=z_to_z_activation,
                             name="x_to_z_lstm")
        z_output = rnn_model(masked_y_input)

        x_input = Input(shape=(timesteps, x_dim), name="x_input")
        masked_x_input = mask(x_input)
        # f(Wz_t+Bx_t), f=identity for now
        xz_input = concatenate([z_output, masked_x_input])
        # mask=Masking(mask_value=0.0, input_shape=(timesteps, y_dim), name="mask")
        xz_output = TimeDistributed(Dense(x_dim, activation=xz_to_y_activation, use_bias=xz_bias), name="xz_to_y_output")(
            xz_input)

        # g(Ay_(t-1)+c), g=identity for now
        y_output = TimeDistributed(Dense(y_dim, activation=y_to_y_activation, use_bias=y_bias), name="y_to_y_output")(
            y_input)
        # f(Wz_t+Bx_t)+g(Ay_(t-1)+c)
        xz_y_input = add([xz_output, y_output])

        # softmax(f(Wz_t+Bx_t)+g(Ay_(t-1)+g))
        main_output = Activation(out_activation)(xz_y_input)
        self.model = Model(inputs=[y_input, x_input], outputs=main_output)


if __name__ == "__main__":
    # random example to test model masking
    import preprocessor

    classes = 4
    train_seqs = [[3, 1, 0, 2, 3],
                  [3, 1, 2, 2],
                  [3, 1]]
    vocab = dict(zip(range(classes), range(classes)))
    xs_train = [[[0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1]],
                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]]]

    prep_bline = preprocessor.BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train_bline, y_train_bline = prep_bline.transform_data(train_seqs, xs=xs_train)
    print x_train_bline
    print "Test baseline:"
    print x_train_bline.shape
    baseline = RNNBaseline(x_train_bline.shape[1], x_train_bline.shape[2], len(vocab), model_name="baseline_model",
                           rnn_type='simpleRNN',
                           loss='categorical_crossentropy', metrics=[], z_activation="relu", z_dim=2)
    baseline.fit_model(x_train_bline, y_train_bline, validation_data=None, n_epochs=20, batch_size=10, verbose=0)
    print baseline.predict(x_train_bline, verbose=True)


    prep = preprocessor.FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train, y_train, xs_train = prep.transform_data(train_seqs, xs=xs_train)
    print "Test full model:"
    full_model = RNNY2YModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=4, z_dim=10,
                             model_name="full_model", rnn_type='simpleRNN',
                             loss='categorical_crossentropy', metrics=[], z_to_z_activation="relu",
                             y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False)
    full_model.fit_model([x_train, xs_train], y_train, validation_data=None, n_epochs=20, batch_size=10, verbose=1)
    print x_train.shape,xs_train.shape
    print full_model.predict([x_train, xs_train],verbose=True)
