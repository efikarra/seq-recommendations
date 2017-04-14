'''
Created on 14 Apr 2017

@author: efi
'''
from main.preprocessor import *
from main.model import *
from main.datasets import *


def run_baseline(train_pois, val_pois, xs_train, xs_val, vocab, pathtosave="model/", filetoread=None):
    preprocessor = BaselinePreprocessor(vocab=vocab,  pad_value=0., seq_length=None)
    X_train, Y_train = preprocessor.transform_data(train_pois, xs=xs_train)
    X_val, Y_val = preprocessor.transform_data(val_pois, xs=xs_val)
    # X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs)
    print "(train sequences,train timesteps,input dimension):", X_train.shape
    print "(val sequences,val timesteps,val dimension):", X_val.shape
    baseline = RNNBaseline(X_train.shape[1], X_train.shape[2], len(vocab), model_name="baseline_model", rnn_type='LSTM',
                           loss='categorical_crossentropy', metrics=[], activation="relu", n_units=2)
    if not filetoread:
        baseline.fit_model(X_train, Y_train, X_val, Y_val, n_epochs=40, batch_size=10, verbose=1)
        baseline.save_model_weights(pathtosave)
    else:
        baseline.load_model_weights(filetoread)
    print baseline.get_layer_weights(2)


def run_fullmodel(train_pois, val_pois, xs_train, xs_val, vocab, pathtosave="model/", filetoread=None):
    preprocessor = FullModelPreprocessor(vocab=vocab, cs_vocab=None, pad_value=0., seq_length=None)
    X_train, Y_train, Xs_train = preprocessor.transform_data(train_pois, xs=xs_train)
    X_val, Y_val, Xs_val = preprocessor.transform_data(val_pois, xs=xs_val)
    # X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs,cs=None)
    print "(train sequences,train timesteps,input dimension):", X_train.shape
    print "(val sequences,val timesteps,val dimension):", X_val.shape
    #     full_model=RNNBXModel(timesteps=X_train.shape[1], features=X_train.shape[2], n_classes=len(vocab), model_name="full_model", rnn_type='LSTM',
    #                  loss='categorical_crossentropy', metrics=[], activation="relu", n_units=10,cv_features=Xs_train.shape[2],cv_units=Xs_train.shape[2],cv_activation="linear")
    full_model = RNNY2YModel(timesteps=X_train.shape[1], x_dim=X_train.shape[2], y_dim=len(vocab),
                             model_name="full_model", rnn_type='simpleRNN',
                             loss='categorical_crossentropy', metrics=[], x_to_y=False, z_to_z_activation="relu",
                             z_dim=10, y_to_y_activation="linear", xz_bias=True)


#     if not filetoread:
#         full_model.fit_model([X_train,Xs_train], Y_train, [X_val,Xs_val],Y_val,n_epochs=20, batch_size=10, verbose=1)
#         full_model.save_model_weights(pathtosave)
#     else:
#         full_model.load_model_weights(filetoread)
# print baseline.get_layer_weights(2)
if __name__ == "__main__":
    flickr_data = load_flickr_data()
    print flickr_table_statistics(flickr_data)
    flickr_data = clean_flickr_data(flickr_data, min_seq_length=1)
    train_table, val_table, test_table = split_flickr_train_val_table(flickr_data, train_percent=0.7, val_percent=0.3,
                                                                      test_percent=0.0)
    seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test, vocab = build_flickr_train_val_seqs(train_table,
                                                                                                    val_table,
                                                                                                    test_table)
    # run_baseline(train_pois,val_pois,xs_train=xs_train,xs_val=xs_val,vocab=vocab,filetoread=None)
    # run_baseline(train_pois,val_pois,xs_train=xs_train,xs_val=xs_val,vocab=vocab,filetoread="model/baseline_model.h5")
    run_fullmodel(seqs_train, seqs_val, xs_train, xs_val, vocab)
