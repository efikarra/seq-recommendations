from model import RNNBaseline, RNNY2YModel
from preprocessor import FullModelPreprocessor, BaselinePreprocessor
import datasets
import sampler
import numpy as np


def create_baseline(timesteps, features, n_classes, model_name="baseline_model", rnn_type='simpleRNN',
                    z_activation="relu", z_dim=2):
    # initialize model
    baseline = RNNBaseline(timesteps, features, n_classes, model_name, rnn_type, z_activation=z_activation, z_dim=z_dim)
    return baseline


def prepare_fullmodel_input_data(seqs_train, seqs_val, xs_train, xs_val, vocab, full_model=True):
    preprocessor = FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train, y_train, train_xs = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val, val_xs = preprocessor.transform_data(seqs_val, xs=xs_val)
    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    return x_train, y_train, train_xs, x_val, y_val, val_xs


def create_fullmodel(timesteps, x_dim, y_dim, z_dim=10, model_name="full_model", rnn_type='simpleRNN',
                     z_to_z_activation="relu",
                     y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False, y_to_y_trainable=True,
                     y_to_y_weights=None):
    # initialize model
    full_model = RNNY2YModel(timesteps=timesteps, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                             model_name=model_name, rnn_type=rnn_type, z_to_z_activation=z_to_z_activation,
                             y_to_y_activation=y_to_y_activation, y_bias=y_bias, z_bias=z_bias, xz_bias=xz_bias)
    # set non trainable weights
    full_model.set_layer_weights_trainable("y_to_y_output", trainable=y_to_y_trainable)
    print "full model summary:"
    full_model.model.summary()
    trainable_weights, non_trainable_weights = full_model.get_model_weights()
    print "trainable weights: %s, non trainable weights %s" % (trainable_weights, non_trainable_weights)
    # initialize weights
    if y_to_y_weights is not None:
        full_model.set_layer_weights("y_to_y_output", y_to_y_weights)
    return full_model


def prepare_baseline_input_data(seqs_train, seqs_val, xs_train, xs_val, vocab):
    preprocessor = BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=None)
    x_train, y_train = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val = preprocessor.transform_data(seqs_val, xs=xs_val)
    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    return x_train, y_train, x_val, y_val


# print baseline.get_layer_weights(2)

def run_model(model, x_train, y_train, x_val, y_val, loss='categorical_crossentropy', metrics=[], optimizer='adam',
              n_epochs=20, batch_size=10, verbose=1, dir_save="trained_model/",
              read_file=None):
    if not read_file:
        history=model.fit_model(x_train, y_train, validation_data=(x_val, y_val), loss=loss, metrics=metrics,
                        optimizer=optimizer, n_epochs=n_epochs, plot_history=True,
                        batch_size=batch_size, verbose=verbose)
        model.save_model_weights(dir_save)
    else:
        model.load_model_weights(read_file)
    val_losses = []
    train_losses = []
    for key, value in history.history.items():
        if key == "val_loss":
            val_losses = value
        if key == "loss":
            train_losses = value

    print "min val loss: %f at epoch: %d"%(np.min(val_losses),np.argmin(val_losses))
    print "train loss: %f at epoch: %d"%(train_losses[np.argmin(val_losses)],np.argmin(val_losses))
    return history

def inspect_weights(model, layer_names):
    for name in layer_names:
        print model.get_layer_weights(name)


if __name__ == "__main__":
    flickr_df = datasets.load_flickr_data()
    print datasets.flickr_table_statistics(flickr_df)
    flickr_df = datasets.clean_flickr_data(flickr_df, min_seq_length=1)

    train_table, val_table, test_table = datasets.split_flickr_train_val_df(flickr_df, train=0.7, val=0.3,
                                                                            test=0.0)
    seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test, vocab = datasets.build_flickr_train_val_seqs(train_table, val_table, test_table)
    alpha, gamma = sampler.transition_matrix(seqs_train, vocab, k=1.0, end_state=False)

    # for baseline model: prepare data, create model and run
    x_train, y_train, x_val, y_val = prepare_baseline_input_data(seqs_train, seqs_val, xs_train=xs_train, xs_val=xs_val, vocab=vocab)
    baseline = create_baseline(timesteps=x_train.shape[1], features=x_train.shape[2], n_classes=len(vocab),
                               model_name="baseline_model_with_xs", rnn_type='LSTM', z_activation="relu", z_dim=10)
    history=run_model(baseline, x_train, y_train, x_val, y_val, loss='categorical_crossentropy', metrics=[], optimizer='adam', n_epochs=50, batch_size=10, verbose=1, dir_save="trained_model/",
              read_file=None)

    # for full model: prepare data, create model and run
    # x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_fullmodel_input_data(seqs_train, seqs_val, xs_train,
    #                                                                                 xs_val, vocab, full_model=True)
    # y_to_y_trainable = True
    # y_to_y_weights = [np.reshape(alpha, (len(alpha), len(alpha)))]
    # y_to_y_weights=None
    # full_model = create_fullmodel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), z_dim=10, rnn_type='LSTM',z_to_z_activation="relu",
    #                               y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False, y_to_y_trainable=y_to_y_trainable, y_to_y_weights=None)

    # run_model(full_model, [x_train, train_xs], y_train, [x_val, val_xs], y_val, n_epochs=50, batch_size=10, verbose=1,
    #           dir_save="trained_model/", read_file=None)

    # y_to_y_weights = full_model.get_layer_weights("y_to_y_output")
    # xz_weights= full_model.get_layer_weights("xz_to_y_output")