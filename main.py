from model import RNNBaseline, RNNY2YModel
from preprocessor import FullModelPreprocessor, BaselinePreprocessor
import datasets
import sampler


def run_baseline(seqs_train, seqs_val, xs_train, xs_val, vocab, dirtosave="trained_model/", filetoread=None):
    preprocessor = BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train, y_train = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val = preprocessor.transform_data(seqs_val, xs=xs_val)
    # X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs)
    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    baseline = RNNBaseline(x_train.shape[1], x_train.shape[2], len(vocab), model_name="baseline_model", rnn_type='simpleRNN',
                           loss='categorical_crossentropy', metrics=[], z_activation="relu", z_dim=2)
    if not filetoread:
        baseline.fit_model(x_train, y_train, validation_data=(x_val, y_val), n_epochs=40, batch_size=10, verbose=1)
        baseline.save_model_weights(dirtosave)
    else:
        baseline.load_model_weights(filetoread)
    #print baseline.get_layer_weights(2)


def run_fullmodel(seqs_train, seqs_val, xs_train, xs_val, vocab, dirsave="trained_model/", filetoread=None):
    preprocessor = FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train, y_train, train_xs = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val, val_xs = preprocessor.transform_data(seqs_val, xs=xs_val)
    # X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs,cs=None)
    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    full_model = RNNY2YModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab),z_dim=10,
                             model_name="full_model", rnn_type='simpleRNN',
                             loss='categorical_crossentropy', metrics=[], z_to_z_activation="relu",
                             y_to_y_activation="linear",  y_bias=False, z_bias=True,xz_bias=False)


    if not filetoread:
        full_model.fit_model([x_train,train_xs], y_train, validation_data=([x_val,val_xs],y_val),n_epochs=20, batch_size=10, verbose=1)
        full_model.save_model_weights(dirsave)
    else:
        full_model.load_model_weights(filetoread)
# print baseline.get_layer_weights(2)

if __name__ == "__main__":
    flickr_data = datasets.load_flickr_data()
    print datasets.flickr_table_statistics(flickr_data)
    flickr_data = datasets.clean_flickr_data(flickr_data, min_seq_length=1)
    #alpha, gamma = sampler.transition_matrix(seqs, vocab, k)
    train_table, val_table, test_table = datasets.split_flickr_train_val_df(flickr_data, train=0.7, val=0.3,
                                                                   test=0.0)
    seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test, vocab = datasets.build_flickr_train_val_seqs(train_table,
                                                                                                    val_table,
                                                                                                    test_table)
    run_baseline(seqs_train,seqs_val,xs_train=xs_train,xs_val=xs_val,vocab=vocab,filetoread=None)
    run_fullmodel(seqs_train, seqs_val, xs_train, xs_val, vocab)
