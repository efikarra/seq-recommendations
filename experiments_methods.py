"""
Helper methods for experiments
"""
from model import RNNBaseline, RNNFullModel, ValLossHistoryCut, MultinomialModel, MarkovModel, NoRecurrenceModel
from preprocessor import FullModelPreprocessor, BaselinePreprocessor
import numpy as np
import utils
from keras.callbacks import EarlyStopping


def prepare_baseline_input(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, wrt_time=False):
    preprocessor = BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=max_seq_length)
    # hack here when we split w.r.t for the validation data.
    # important to keep the following if-then=else
    x_train, y_train = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val = preprocessor.transform_data(seqs_val, xs=xs_val)

    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    return x_train, y_train, x_val, y_val


def prepare_fullmodel_input(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, split_wrt_time=False):
    preprocessor = FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=max_seq_length)
    # hack here when we split w.r.t for the validation data.
    # important to keep the following if-then=else
    x_train, y_train, train_xs = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val, val_xs = preprocessor.transform_data(seqs_val, xs=xs_val)
    print "(train sequences,train timesteps,input dimension):", x_train.shape
    print "(val sequences,val timesteps,val dimension):", x_val.shape
    return x_train, y_train, train_xs, x_val, y_val, val_xs


def run_model(model, x_train, y_train, validation_data=None, orig_seqs_lengths=None, loss='categorical_crossentropy',
              metrics=[], optimizer='adam',
              n_epochs=20, batch_size=10, verbose=1, dir_save="trained_models/", early_stopping=False, wrt_time=False,
              read_file=None):
    callbacks = []
    rnn_validation_data = validation_data
    if wrt_time:
        val_history = ValLossHistoryCut(validation_data, orig_seqs_lengths)
        rnn_validation_data = None
        callbacks.append(val_history)
    if early_stopping:
        if wrt_time:
            monitor="my_loss"
        else:
            monitor="val_loss"
        stopping = EarlyStopping(monitor=monitor, min_delta=0, patience=0, verbose=0, mode='auto')
        callbacks.append(stopping)

    history = None
    if not read_file:
        history = model.fit_model(x_train, y_train, validation_data=rnn_validation_data, loss=loss, metrics=metrics,
                                  optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                                  callbacks=callbacks)
        model.save_model_weights(dir_save)
    else:
        model.load_model_weights(read_file)

    if wrt_time:
        history.history['val_loss'] = val_history.val_lossses

    return history


def analyze_history(history, filepath):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    for key, value in history.history.items():
        print key, value
    utils.plot(values=[train_losses, val_losses], colors=['b', 'g'],
               labels=['Train loss', 'Val loss'], ylabel='loss', xlabel='epoch', save_path=filepath + "_loss")
    print "min val loss: %f at epoch: %d" % (np.min(val_losses), np.argmin(val_losses)+1)
    print "train loss: %f at epoch: %d" % (train_losses[np.argmin(val_losses)], np.argmin(val_losses)+1)


def run_multinomial(seqs_train, val_seqs, n_classes, wrt_time=False):
    mult_model = MultinomialModel(n_classes, model_name="multi_model", k=1.0)
    mult_model.fit_model(seqs_train)
    val_preds = mult_model.predict(val_seqs)
    if wrt_time:
        train_neg_ll, val_neg_ll = utils.compute_likelihood_cut(val_preds, 0.7, count_first_prob=False)
    else:
        train_preds=mult_model.predict(seqs_train)
        train_neg_ll= utils.compute_likelihood(train_preds, count_first_prob=False)
        val_neg_ll= utils.compute_likelihood(val_preds, count_first_prob=False)
    print "Multinomial train neg ll: %f, val neg ll: %f" % (train_neg_ll, val_neg_ll)


def run_markov(seqs_train, val_seqs, n_classes, wrt_time=False):
    markov_model = MarkovModel(n_classes, model_name="markov_model", order=1, k=1.0)
    markov_model.fit_model(seqs_train)
    val_preds = markov_model.predict(val_seqs)
    if wrt_time:
        train_neg_ll, val_neg_ll = utils.compute_likelihood_cut(val_preds, 0.7, count_first_prob=False)
    else:
        train_preds = markov_model.predict(seqs_train)
        train_neg_ll = utils.compute_likelihood(train_preds, count_first_prob=False)
        val_neg_ll = utils.compute_likelihood(val_preds, count_first_prob=False)
    print "Markov train neg ll: %f, val neg ll: %f" % (train_neg_ll, val_neg_ll)
    return markov_model


def run_baseline_rnn(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, orig_seqs_lengths=None, with_xs=True, wrt_time=False,
                     read_file=None, rnn_type="LSTM", z_dim=10, z_activation="relu", early_stopping=False, n_epochs=50, verbose=1,
                     batch_size=10, model_name="test_model"):
    """Runs a baseline RNN model with connections: [y_(t-1),x_t]->z_t and z_t->y_t.
       NO direct connections between y's. NO direct connections of x's to y's. 
       Set with_xs to False to remove x's from input features."""
    if with_xs:
        x_train, y_train, x_val, y_val = prepare_baseline_input(seqs_train, seqs_val, xs_train=xs_train, xs_val=xs_val,
                                                            vocab=vocab, max_seq_length=max_seq_length, wrt_time=wrt_time)
    else:
        x_train, y_train, x_val, y_val = prepare_baseline_input(seqs_train, seqs_val, xs_train=None, xs_val=None,
                                                                vocab=vocab, max_seq_length=max_seq_length, wrt_time=wrt_time)
    baseline = RNNBaseline(x_train.shape[1], x_train.shape[2], len(vocab), model_name=model_name,
                           rnn_type=rnn_type, z_activation=z_activation, z_dim=z_dim)
    print
    print baseline.model_name
    print baseline.model.summary()
    history = run_model(baseline, x_train, y_train, validation_data=(x_val, y_val), orig_seqs_lengths=orig_seqs_lengths,
                        loss='categorical_crossentropy', metrics=[], optimizer='adam', n_epochs=n_epochs, batch_size=batch_size,
                        verbose=verbose, early_stopping=early_stopping, wrt_time=wrt_time, read_file=read_file)
    analyze_history(history, baseline.model_name)
    return baseline


def run_rnn_xtoy(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, orig_seqs_lengths=None, wrt_time=False,
                 read_file=None, rnn_type="LSTM", early_stopping=False, n_epochs=50, batch_size=10,
                 verbose=1, model_name="test_model"):
    """Runs an RNN model with connections: y_(t-1)->z_t, x_t->y_t and z_t->y_t. No direct connections between y's.
        """
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_fullmodel_input(seqs_train, seqs_val, xs_train,
                                                                               xs_val, vocab,
                                                                               max_seq_length=max_seq_length, split_wrt_time=wrt_time)
    full_model = RNNFullModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), z_dim=10,
                              model_name=model_name, rnn_type=rnn_type, z_to_z_activation="relu",
                              y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False,
                              y_to_y_regularizer=None,
                              connect_x_to_y=True, connect_y_to_y=False)
    print full_model.model_name
    print full_model.model.summary()
    history = run_model(full_model, [x_train,train_xs], y_train, validation_data=([x_val,val_xs], y_val),
                        orig_seqs_lengths=orig_seqs_lengths, loss='categorical_crossentropy', metrics=[],
                        optimizer='adam',
                        n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                        early_stopping=early_stopping, wrt_time=wrt_time, read_file=read_file)
    analyze_history(history, full_model.model_name)


def run_rnn_only_ytoy(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, orig_seqs_lengths=None, wrt_time=False,
                      read_file=None, early_stopping=False, n_epochs=50, batch_size=10,
                      verbose=1, model_name="test_model", connect_x=True, connect_y=True, y_bias=True, embed_y=False, z_dim=10,):
    """Runs an RNN model with connections: y_(t-1)->z_t, x_t->y_t and z_t->y_t. No direct connections between y's.
        """
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_fullmodel_input(seqs_train, seqs_val, xs_train,
                                                                               xs_val, vocab, max_seq_length=max_seq_length, split_wrt_time=wrt_time)

    full_model = NoRecurrenceModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), model_name=model_name,
                                   y_to_y_activation="linear", x_to_y_activation="linear", out_activation="softmax",
                                   y_bias=y_bias, xy_bias=False, y_to_y_regularizer=None, connect_x=connect_x, connect_y=connect_y, embed_y=embed_y, z_dim=z_dim)
    print full_model.model_name
    print full_model.model.summary()
    if connect_x:
        if connect_y:
            train=[x_train,train_xs]
            validation=[x_val,val_xs]
        else:
            train = train_xs
            validation = val_xs
    else:
        train=x_train
        validation=x_val
    history = run_model(full_model, train, y_train, validation_data=(validation, y_val),
                        orig_seqs_lengths=orig_seqs_lengths, loss='categorical_crossentropy', metrics=[],
                        optimizer='adam',
                        n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                        early_stopping=early_stopping, wrt_time=wrt_time, read_file=read_file)
    analyze_history(history, full_model.model_name)
    return full_model


def run_rnn_ytoy(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, orig_seqs_lengths=None, y_to_y_trainable=True, y_to_y_weights=None,
                 y_to_y_regularizer=None, wrt_time=False, read_file=None, rnn_type="LSTM", early_stopping=False, n_epochs=50, verbose=1,
                 batch_size=10, model_name="test_model"):
    """Runs a RNN model with connections: x_t->z_t, y_(t-1)->y_t and z_t->y_t.
           Direct connections between y's. NO direct connections of x's to y's. 
           """
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_fullmodel_input(seqs_train, seqs_val, xs_train,
                                                                               xs_val, vocab, max_seq_length=max_seq_length, split_wrt_time=wrt_time)
    full_model = RNNFullModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), z_dim=10,
                              model_name=model_name, rnn_type=rnn_type, z_to_z_activation="relu",
                              y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False,
                              y_to_y_regularizer=y_to_y_regularizer,
                              connect_x_to_y=False, connect_y_to_y=True)
    full_model.set_layer_weights_trainable("y_to_y_output", trainable=y_to_y_trainable)
    trainable_weights, non_trainable_weights = full_model.get_model_weights()
    print "trainable weights: %s, non trainable weights %s" % (trainable_weights, non_trainable_weights)
    if y_to_y_weights is not None:
        full_model.set_layer_weights("y_to_y_output", y_to_y_weights)
    print full_model.model_name
    print full_model.model.summary()
    history = run_model(full_model, [x_train,train_xs], y_train, validation_data=([x_val,val_xs], y_val),
                        orig_seqs_lengths=orig_seqs_lengths, loss='categorical_crossentropy', metrics=[],
                        optimizer='adam',
                        n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                        early_stopping=early_stopping, wrt_time=wrt_time, read_file=read_file)
    analyze_history(history, full_model.model_name)


def run_rnn_fullmodel(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length, orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True, y_to_y_weights=None,
                      y_to_y_regularizer=None, read_file=None, rnn_type="LSTM", early_stopping=False, n_epochs=50,
                      batch_size=10, verbose=1, z_dim=10, model_name="test_model"):
    """Runs a RNN model with connections: y_(t-1)]->z_t, y_(t-1)->y_t, x_t->y_t and z_t->y_t.
       So, y's are directly connected. Also, x's are directly connected to y's. """
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_fullmodel_input(seqs_train, seqs_val, xs_train,
                                                                               xs_val, vocab, max_seq_length=max_seq_length)
    full_model = RNNFullModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), z_dim=z_dim,
                              model_name=model_name, rnn_type=rnn_type, z_to_z_activation="relu",
                              y_to_y_activation="linear", y_bias=False, z_bias=True, xz_bias=False,
                              y_to_y_regularizer=y_to_y_regularizer,
                              connect_x_to_y=True, connect_y_to_y=True)
    full_model.set_layer_weights_trainable("y_to_y_output", trainable=y_to_y_trainable)
    trainable_weights, non_trainable_weights = full_model.get_model_weights()
    print "trainable weights: %s, non trainable weights %s" % (trainable_weights, non_trainable_weights)
    if y_to_y_weights is not None:
        full_model.set_layer_weights("y_to_y_output", y_to_y_weights)
    print full_model.model_name
    print full_model.model.summary()
    history = run_model(full_model, [x_train,train_xs], y_train, validation_data=([x_val,val_xs], y_val),
                        orig_seqs_lengths=orig_seqs_lengths, loss='categorical_crossentropy', metrics=[],
                        optimizer='adam',
                        n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                        early_stopping=early_stopping, wrt_time=wrt_time, read_file=read_file)
    analyze_history(history, full_model.model_name)


def get_model_weights(model, layer_name):
    weights_list=[]
    weights = model.get_layer_weights(layer_name)
    weights_list.append(weights)
    return weights_list
