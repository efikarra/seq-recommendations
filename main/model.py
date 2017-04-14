'''
Created on 5 Apr 2017

@author: efi
'''
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Input,merge,Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Masking
import matplotlib.pyplot as plt
from preprocessor import *
from keras.layers.wrappers import TimeDistributed
from keras.layers import Merge
import h5py

class BaseModel():
    def __init__(self, n_classes, model_name="test_model", rnn_type='simpleRNN', loss='categorical_crossentropy',
                 metrics=[]):
        self.model_name = model_name
        self.rnn_type = rnn_type
        self.loss = loss
        self.metrics = metrics
        self.n_classes = n_classes
        self.model = None
    def fit_model(self, X_train, Y_train, X_val, Y_val, n_epochs=10, batch_size=100, plot_history=False, verbose=1):
        history = self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=n_epochs,
                                 batch_size=batch_size, verbose=verbose)
        # print fileprefix+": "+self.evaluate(X_test,Y_test)
        for key, value in history.history.items():
            print key, value
        if plot_history:
            self.plot_history(history)

    def predict_classes(self, X_test, batch_size=10, verbose=1):
        return self.model.predict_classes(X_test, verbose=verbose)

    def predict_proba(self, X_test, verbose=False):
        return self.model.predict_proba(X_test, verbose=verbose)

    def evaluate(self, X_test, Y_test):
        scores = self.model.evaluate(X_test, Y_test)
        return (self.model.metrics_names, scores)

    def plot_history(self, history):
        plt.plot(history.history['loss'], color='b', label='Train loss')
        plt.plot(history.history['val_loss'], color='g', label='Test loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig(self.model_name + 'loss')
        plt.close()


    def save_model_weights(self, filepath):
        file = h5py.File(filepath + self.model_name + ".h5",'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight'+str(i),data=weight[i])
        file.close()
        print("Saved model weights to disk: " + filepath + self.model_name)

    def load_model_weights(self, filepath):
        # workaround to load weights into new model
        file=h5py.File(filepath,'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight'+str(i)][:])
        self.model.set_weights(weight)
    def get_layer_weights(self,index):
        return self.model.layers[index].get_weights()
class RNNBaseline(BaseModel):
    def __init__(self, timesteps, features, n_classes, model_name="baseline_model", rnn_type='simpleRNN',
                 loss='categorical_crossentropy', metrics=[], activation="relu", n_units=20):
        BaseModel.__init__(self, n_classes, model_name=model_name, rnn_type=rnn_type, loss=loss,
                          metrics=metrics)
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(timesteps, features), name="mask"))
        if self.rnn_type == 'simpleRNN':
            self.model.add(
                SimpleRNN(n_units, input_shape=(timesteps, features), return_sequences=True,activation=activation, name="rnn"))
        
        if self.rnn_type == 'LSTM':
            self.model.add(
                LSTM(n_units, input_shape=(timesteps, features), return_sequences=True,activation=activation, name="lstm"))

        self.model.add(TimeDistributed(Dense(n_classes, activation='softmax'), name="output"))
        self.model.compile(loss=self.loss, optimizer='adam', metrics=[self.loss] + self.metrics)
        print "baseline model summary:"
        print self.model.summary()
class RNNY2YModel(BaseModel):
    def __init__(self, timesteps, x_dim, y_dim, model_name="y_to_y_model", rnn_type='simpleRNN',
                 loss='categorical_crossentropy', metrics=[],x_to_y=False, z_to_z_activation="relu", z_dim=20,y_to_y_activation="linear",xz_to_y_activation="linear",y_bias=False,xz_bias=False):
        BaseModel.__init__(self, y_dim, model_name=model_name, rnn_type=rnn_type, loss=loss,
                          metrics=metrics)

        y_input=Input(shape=(timesteps,y_dim),name="y_input") 
        x_input=Input(shape=(timesteps,x_dim),name="x_input")
        
        #build rnn model
        rnn_model = Sequential()
        rnn_model.add(Masking(mask_value=0.0, input_shape=(timesteps, x_dim), name="mask"))
        if self.rnn_type == 'simpleRNN':
            rnn_model.add(
                SimpleRNN(z_dim, input_shape=(timesteps, x_dim), return_sequences=True,activation=z_to_z_activation, name="rnn-x-to-z"))
        
        if self.rnn_type == 'LSTM':
            rnn_model.add(
                LSTM(z_dim, input_shape=(timesteps, x_dim), return_sequences=True,activation=z_to_z_activation, name="lstm-x-to-z"))
        z_output=rnn_model(x_input)
        print "rnnmodel summary:"
        print rnn_model.summary()
        xz_output=z_output
        if x_to_y:
            # f(Wz_t+Bx_t), f=identity for now
            xz_input = merge([z_output, x_input], mode='concat')
            xz_output=TimeDistributed(Dense(x_dim, activation=xz_to_y_activation,bias=xz_bias), name="xz_output")(xz_input)
        
        # g(Ay_(t-1)+c), g=identity for now
        y_output=TimeDistributed(Dense(y_dim, activation=y_to_y_activation,bias=y_bias), name="y_output")(y_input)
        # f(Wz_t+Bx_t)+g(Ay_(t-1)+c)
        if x_to_y:
            xz_y_input=merge([xz_output,y_output], mode='sum')
        else:
            xz_y_input=merge([xz_output,y_output], mode='concat')
        
        # softmax(f(Wz_t+Bx_t)+g(Ay_(t-1)+g))
        main_output=Activation("softmax")(xz_y_input)
        
        self.model=Model(inputs=[x_input, y_input], outputs=main_output)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=[self.loss] + self.metrics)
        print "full model summary:"
        print self.model.summary()