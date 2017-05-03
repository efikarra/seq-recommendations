"""
Models implementation.
"""
import h5py
import os
import numpy as np
from keras.layers import Input, Activation
from keras.layers.core import Dense, Masking, Dropout
from keras.layers.merge import concatenate, add
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential
import utils
from keras.callbacks import Callback
from keras import backend as K
from keras.regularizers import Regularizer
from keras.constraints import Constraint,max_norm
from keras.initializers import Initializer
from keras.regularizers import l2
from theano import tensor


class ModelResults():
    def __init__():
        pass
    def __init__(self,train_loss=None,val_loss=None,epoch=None):
        self.val_loss=val_loss
        self.train_loss=train_loss
        self.epoch = epoch


class ArrayInitializer(Initializer):
    """Initializer that generates tensors initialized to a constant value.
    # Arguments
        value: float; the value of the generator tensors.
    """

    def __init__(self, values=0):
        self.values = values

    def __call__(self, shape, dtype=None):
        return self.values

    def get_config(self):
        return {'value': self.values}


class OnlyNonZeroDiagonal(Constraint):
    """Constrains the weight matrix to have only the diagonal non-zero.
    """

    def __init__(self, input_dim, skip_cols):
        self.input_dim = input_dim
        self.skip_cols=skip_cols

    def __call__(self, w):
        if self.skip_cols>0:
            mask =K.concatenate([K.ones((self.skip_cols,self.input_dim)),K.eye(self.input_dim)], axis=0)
        else:
            mask = K.eye(self.input_dim)
        w *=K.cast(mask,K.floatx())
        return w

    def get_config(self):
        return {'input_dim': self.input_dim,
                'skip_cols': self.skip_cols}

only_non_zero_diag = OnlyNonZeroDiagonal


class GaussPriorRegularizer(Regularizer):
    def __init__(self, means, var):
        from theano.tensor import _shared
        self.means = _shared(means.astype("float32"))
        self.var = var

    def __call__(self, x):
        regularization = 0.

        strength=K.cast_to_floatx(1/(2*self.var))
        print 1/(2*self.var)
        regularization += K.sum(strength * K.square(x-self.means))
        return regularization

    def get_config(self):
        return {'var': float(self.strength),
                'means': self.means}


def gauss_prior(means, var):
    return GaussPriorRegularizer(means, var)


class ValLossHistoryCut(Callback):
    def __init__(self, val_data, orig_seqs_lengths):
        Callback.__init__(self)
        self.val_data = val_data
        self.orig_seqs_lengths = orig_seqs_lengths

    def on_train_begin(self, logs={}):
        self.val_lossses = []
        if "my_loss" not in logs:
            logs["my_loss"] = 0.0
        return

    def on_epoch_end(self, epoch, logs={}):
        y_perds = self.model.predict(self.val_data[0])
        m = np.multiply(y_perds, self.val_data[1])
        y_perds2 = np.max(m, axis=2)
        y_perds2 = K.clip(y_perds2, K.epsilon(), 1.0 - K.epsilon())
        train_neg_ll, val_neg_ll = utils.compute_likelihood_cut(K.eval(y_perds2), 0.7,
                                                                orig_lengths=self.orig_seqs_lengths)
        self.val_lossses.append(val_neg_ll)
        logs["my_loss"] = val_neg_ll
        print "\n"
        print "my val loss:", val_neg_ll
        return


class BaseModel():
    def __init__(self, n_classes, model_name="test_model"):
        self.n_classes = n_classes
        self.model_name = model_name
        self.model = None


class MultinomialModel(BaseModel):
    def __init__(self, n_classes, model_name="multinomial_model", k=1.0):
        BaseModel.__init__(self, n_classes, model_name)
        self.model = np.zeros((1, self.n_classes))
        self.k = k

    def fit_model(self, seqs, normalize=True):
        self.model = utils.multinomial_probabilities(seqs, self.n_classes, self.k, normalize)
        if normalize:
            assert np.isclose(np.sum(self.model,axis=1),self.model.shape[0]), "ERROR: multinomial probabilities do not sum to 1!"

    def predict(self, seqs):
        predictions = []
        for seq in seqs:
            pred = [self.model[0, s] for s in seq]
            predictions.append(pred)
        return predictions


class MarkovModel(BaseModel):
    def __init__(self, n_classes, model_name="markov_model", order=1, k=1.0):
        BaseModel.__init__(self, n_classes, model_name)
        assert order == 1, "ERROR: only first-order Markov chains are supported for now."
        self.model = np.zeros((self.n_classes, self.n_classes))
        self.initial_probs = np.zeros(self.n_classes)
        self.order = order
        self.k = k

    def fit_model(self, seqs, freq=False):
        self.model, self.initial_probs = utils.transition_matrix(seqs, self.n_classes, self.k, freq=freq,
                                                                 end_state=False)

    def predict(self, seqs):
        predictions = []
        for seq in seqs:
            pred = [self.initial_probs[seq[0]]]
            if len(seq) > 1:
                for i, j in zip(seq[:-1], seq[1:]):
                    pred.append(self.model[i, j])
            predictions.append(pred)
        return predictions


class BaseRNNModel(BaseModel):
    def __init__(self, n_classes, model_name="test_model", rnn_type='simpleRNN'):
        BaseModel.__init__(self, n_classes, model_name)
        self.rnn_type = rnn_type

    def compile_model(self, loss='categorical_crossentropy', metrics=[],
                  optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[loss] + metrics)

    def fit_model(self, x_train, y_train, validation_data=None, n_epochs=10, batch_size=100, verbose=1, callbacks=None):

        return self.model.fit(x_train, y_train, validation_data=validation_data, epochs=n_epochs,
                                 batch_size=batch_size, verbose=verbose, callbacks=callbacks)
        # print fileprefix+": "+self.evaluate(X_test,Y_test)

    def fit_generator(self, train_generator, val_generator,steps_per_epoch=50, validation_steps=50,epochs=10):
        return self.model.fit_generator(train_generator, validation_data=val_generator,steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,epochs=epochs)

    def predict(self, x_test, batch_size=10, verbose=1):
        return self.model.predict(x_test, batch_size=batch_size, verbose=verbose)

    def evaluate(self, x_test, y_test, batch_size=10,verbose=0):
        scores = self.model.evaluate(x_test, y_test, verbose=verbose, batch_size=batch_size)
        return self.model.metrics_names, scores

    def save_model_weights(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = h5py.File(directory + self.model_name + ".h5", 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight' + str(i), data=weight[i])
        f.close()
        #print("Saved model weights to disk: " + directory + self.model_name)

    def load_model_weights(self, filepath):
        # workaround to load weights into new model
        # f = h5py.File(filepath, 'r')
        # weight = []
        # for i in range(len(f.keys())):
        #     weight.append(f['weight' + str(i)][:])
        # self.model.set_weights(weight)
        self.model.load_weights(filepath,by_name=False)

    def get_layer_weights(self, layer):
        if isinstance(layer, str):
            return self.model.get_layer(layer).get_weights()
        else:
            return self.model.layers[layer].get_weights()

    def set_layer_weights_trainable(self, name, trainable=True):
        self.model.get_layer(name).trainable = trainable

    def set_layer_weights(self, name, weights):
        self.model.get_layer(name).set_weights(weights)

    def get_model_weights(self):
        return self.model.trainable_weights, self.model.non_trainable_weights

    def get_activations(self, layer, inputs):
        get_activations = K.function([self.model.layers[i].input for i in range(len(inputs))]+[K.learning_phase()], self.model.layers[layer].output)
        activations = get_activations(inputs+[0])
        return activations


class RNNBaseline(BaseRNNModel):
    def __init__(self, timesteps, features, n_classes, model_name="baseline_model", rnn_type='simpleRNN',
                 out_activation="softmax", z_activation="relu", z_dim=20, z_to_y_drop=0.0):
        BaseRNNModel.__init__(self, n_classes, model_name=model_name, rnn_type=rnn_type)
        main_input = Input(shape=(timesteps, features), name='myinput')
        masked_input = Masking(mask_value=0.0, name="mask")(main_input)
        if self.rnn_type == 'simpleRNN':
            rnn_model = SimpleRNN(z_dim,return_sequences=True,
                                  activation=z_activation,name="rnn")


        if self.rnn_type == 'LSTM':
            rnn_model = LSTM(z_dim,  return_sequences=True, activation=z_activation,
                             name='lstm')
        rnn_output = rnn_model(masked_input)
        rnn_output=Dropout(z_to_y_drop)(rnn_output)
        output = TimeDistributed(Dense(n_classes, activation=out_activation), name="output")(rnn_output)
        self.model = Model(inputs=main_input, outputs=output)
        # self.model = Sequential()
        # self.model.add(SimpleRNN(5, input_shape=(timesteps, features), return_sequences=True))
        # self.model.add(TimeDistributed(Dense(n_classes, activation="softmax")))


class NoRecurrenceModel(BaseRNNModel):
    def __init__(self, timesteps, x_dim, y_dim, model_name="y_to_y_model",
                 y_to_y_activation="linear", x_to_y_activation="linear", y_to_y_w_initializer=None, out_activation="softmax",
                 y_bias=False, xy_bias=False, y_to_y_regularizer=None, z_dim=10, z_bias=True, connect_x=True, connect_y=True, embed_y=False, diag_b=True):
        assert "ERROR: the model needs an input! either x or y should be added.", (connect_x == False and connect_y == False)
        BaseRNNModel.__init__(self, y_dim, model_name=model_name, rnn_type=None)
        if y_to_y_w_initializer is None:
            y_to_y_w_initializer='random_uniform'
        if connect_y:
            y_input = Input(shape=(timesteps, y_dim), name="y_input")
            mask1 = Masking(mask_value=0.0, input_shape=(timesteps, y_dim), name="mask1")
            masked_y_input = mask1(y_input)
            if embed_y:
                # z_t=Wy_(t-1)+c, learn embedding of (y_t-1)
                z_output = TimeDistributed(
                    Dense(z_dim, activation="linear", use_bias=z_bias), name="y_to_z_output")(masked_y_input)
            else:
                # z_t=y_(t-1), dirctly connect y_(t-1)
                z_output = masked_y_input

            # Az_(t-1)
            y_output = TimeDistributed(
                Dense(y_dim, activation=y_to_y_activation, kernel_initializer=y_to_y_w_initializer,
                    kernel_regularizer=y_to_y_regularizer,use_bias=y_bias),
                name="y_output")(z_output)

        # f(Bx_t), f=identity for now
        if connect_x:
            x_input = Input(shape=(timesteps, x_dim), name="x_input")
            mask2 = Masking(mask_value=0.0, input_shape=(timesteps, x_dim), name="mask2")
            masked_x_input = mask2(x_input)
            kernel_constraint=None
            if diag_b:
                kernel_constraint=only_non_zero_diag(x_dim,0)
            x_output = TimeDistributed(Dense(y_dim, activation=x_to_y_activation, use_bias=xy_bias
                                             ,kernel_constraint=kernel_constraint),
                                       name="x_to_y_output")(masked_x_input)

            if connect_y:
                # f(Bx_t)+g(Az_(t-1)+c)
                x_y_input = add([x_output, y_output])
            else:
                x_y_input=x_output
        else:
            x_y_input = y_output

        # softmax(f(Bx_t)+g(Ay_(t-1)+g))
        main_output = Activation(out_activation)(x_y_input)
        if connect_x:
            if connect_y:
                inputs=[y_input, x_input]
            else:
                inputs = x_input
        else:
            inputs=y_input
        self.model = Model(inputs=inputs, outputs=main_output)


class RNNFullModel(BaseRNNModel):
    def __init__(self, timesteps, x_dim, y_dim, z_dim=20, model_name="y_to_y_model", rnn_type='simpleRNN',
                 z_to_z_activation="relu",
                 y_to_y_activation="linear", xz_to_y_activation="linear", y_to_y_w_initializer=None, out_activation="softmax",
                 ytoy_bias=False, toy_bias=False, z_bias=True, y_to_y_regularizer=None, toy_regularizer=None,y_to_z=True,
                 y_to_y=True, x_to_y=True, x_to_z=False, z_to_y_drop=0.0, diag_b=True,y_to_z_dropout=0.0):
        BaseRNNModel.__init__(self, y_dim, model_name=model_name, rnn_type=rnn_type)
        assert "ERROR: the model needs an input into z's! either x or y should be added.", (x_to_z == False and y_to_z == False)
        if y_to_y_w_initializer is None:
            y_to_y_w_initializer='glorot_uniform'
        if y_to_y or y_to_z:
            y_input = Input(shape=(timesteps, y_dim), name="y_input")
            mask1 = Masking(mask_value=0.0, input_shape=(timesteps, y_dim), name="mask1")
            masked_y_input = mask1(y_input)

        if x_to_y or x_to_z:
            x_input = Input(shape=(timesteps, x_dim), name="x_input")
            mask2 = Masking(mask_value=0.0, input_shape=(timesteps, x_dim), name="mask2")
            masked_x_input = mask2(x_input)


        if self.rnn_type == 'simpleRNN':
            rnn_model = SimpleRNN(z_dim, input_shape=(timesteps, x_dim), use_bias=z_bias, return_sequences=True,
                                  activation=z_to_z_activation,
                                  name="z_to_z_rnn")

        if self.rnn_type == 'LSTM':
            rnn_model = LSTM(z_dim, input_shape=(timesteps, x_dim), return_sequences=True, use_bias=z_bias,
                             activation=z_to_z_activation,
                             name="z_to_z_lstm")

        if y_to_z and x_to_z:
            z_input=concatenate([masked_y_input, masked_x_input])
            z_input = Dropout(z_to_y_drop)(z_input)
            z_output = rnn_model(z_input)
        else:
            if y_to_z:
                z_input =masked_y_input
                z_input = Dropout(z_to_y_drop)(z_input)
                z_output = rnn_model(z_input)
            elif x_to_z:
                z_input =masked_x_input
                z_input = Dropout(z_to_y_drop)(z_input)
                z_output = rnn_model(z_input)

        z_output = Dropout(z_to_y_drop)(z_output)

        # f(Wz_t+Bx_t), f=identity for now
        xtoy_kernel_constraint = None
        if x_to_y:
            toy_input1 = concatenate([z_output, masked_x_input])
            if diag_b:
                xtoy_kernel_constraint = only_non_zero_diag(x_dim, skip_cols=z_dim)
        else:
            toy_input1 = z_output
        toy_output1 = TimeDistributed(Dense(y_dim, activation=xz_to_y_activation, use_bias=toy_bias,
                                            kernel_regularizer=toy_regularizer,kernel_constraint=xtoy_kernel_constraint),
                                    name="to_y_output")(toy_input1)

        if y_to_y:
            # g(Ay_(t-1)+c), g=identity for now
            toy_output2 = TimeDistributed(
                Dense(y_dim, activation=y_to_y_activation, kernel_regularizer=y_to_y_regularizer,
                      kernel_initializer=y_to_y_w_initializer, use_bias=ytoy_bias),name="y_to_y_output")(y_input)
            # f(Wz_t+Bx_t)+g(Ay_(t-1)+c)
            final_y_input = add([toy_output1, toy_output2])
        else:
            final_y_input = toy_output1

        # softmax(f(Wz_t+Bx_t)+g(Ay_(t-1)+g))
        final_output = Activation(out_activation)(final_y_input)
        inputs=[]
        if y_to_y or y_to_z:
            inputs.append(y_input)
        if x_to_y or x_to_z:
            inputs.append(x_input)
        self.model = Model(inputs=inputs, outputs=final_output)


if __name__ == "__main__":
    pass
    # random example to test model masking
    # import preprocessor
    # import datasets
    #
    # classes = 4
    # sequences = [[3, 1, 0, 2, 3, 2, 3, 1, 3, 2],
    #              [3, 1, 2, 2, 1, 1, 1, 2],
    #              [3, 1, 3, 3, 1]]
    # vocab = dict(zip(range(classes), range(classes)))
    # xs = datasets.build_xs(sequences, vocab)
    # seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test = datasets.split_seqs(sequences, xs, train=0.7, val=0.3,
    #                                                                                  test=0.0)
    #
    # prep_bline = preprocessor.BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    # x_train_bline, y_train_bline = prep_bline.transform_data(seqs_train, xs=xs_train)
    # x_all_bline, y_all_bline = prep_bline.transform_data(sequences, xs=xs, pad=False)
    #
    # print x_train_bline
    # print "Test baseline:"
    # print x_train_bline.shape
    # baseline = RNNBaseline(x_train_bline.shape[1], x_train_bline.shape[2], len(vocab), model_name="baseline_model",
    #                        rnn_type='simpleRNN',
    #                        loss='categorical_crossentropy', metrics=[], z_activation="relu", z_dim=2)
    # my_losses_histories = ValLossHistoryCut(())
    # baseline.fit_model(x_train_bline, y_train_bline, validation_data=None, n_epochs=20, batch_size=10, verbose=0,
    #                    callbacks=[my_losses_histories(())])
    # print(my_losses_histories.val_losses)
    # print(my_losses_histories.train_losses)
    # print baseline.predict(x_train_bline, verbose=True)

    # prep = preprocessor.FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    # x_train, y_train, xs_train = prep.transform_data(train_seqs, xs=xs_train)
    # print "Test full model:"
    # full_model = RNNY2YModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=4, z_dim=10,
    #                          model_name="full_model", rnn_type='simpleRNN',
    #                          loss='categorical_crossentropy', metrics=[], z_to_z_activation="relu",
    #                          y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False)
    # full_model.fit_model([x_train, xs_train], y_train, validation_data=None, n_epochs=20, batch_size=10, verbose=1)
    # print x_train.shape, xs_train.shape
    # print full_model.predict([x_train, xs_train], verbose=True)
