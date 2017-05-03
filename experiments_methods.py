"""
Helper methods for experiments
"""
from model import RNNBaseline, RNNFullModel, ValLossHistoryCut, MultinomialModel, MarkovModel, NoRecurrenceModel, \
    ModelResults
from preprocessor import FullModelPreprocessor, BaselinePreprocessor
import numpy as np
import utils
from keras.callbacks import EarlyStopping,ModelCheckpoint


def prepare_model_input(seqs_train, seqs_val, xs_train, xs_val, vocab, max_seq_length):
    preprocessor = FullModelPreprocessor(vocab=vocab, pad_value=0., seq_length=max_seq_length)
    x_train, y_train, train_xs = preprocessor.transform_data(seqs_train, xs=xs_train)
    x_val, y_val, val_xs = preprocessor.transform_data(seqs_val, xs=xs_val)
    return x_train, y_train, train_xs, x_val, y_val, val_xs


def run_model(model, x_train, y_train, validation_data=None, orig_seqs_lengths=None,model_checkpoint=False,
              n_epochs=20, batch_size=10, verbose=1, dir_save="trained_models/", early_stopping=False, wrt_time=False,
              ):
    callbacks = []
    rnn_validation_data = validation_data
    if wrt_time:
        val_history = ValLossHistoryCut(validation_data, orig_seqs_lengths)
        rnn_validation_data = None
        callbacks.append(val_history)
    if model_checkpoint:
        checkpoint = ModelCheckpoint(dir_save+ model.model_name+".{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoint)
    if early_stopping:
        if wrt_time:
            monitor = "my_loss"
        else:
            monitor = "val_loss"
        stopping = EarlyStopping(monitor=monitor, min_delta=0, patience=0, verbose=0, mode='auto')
        callbacks.append(stopping)

    history = None
    model.compile_model(loss='categorical_crossentropy', metrics=[], optimizer='adam')
    history = model.fit_model(x_train, y_train, validation_data=rnn_validation_data,
                              n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                              callbacks=callbacks)
    if wrt_time:
        if history:
            history.history['val_loss'] = val_history.val_lossses

    return history


def evaluate_full_model(model, seqs_test, xs_test, vocab, max_seq_length, with_xs=True):
    preprocessor = BaselinePreprocessor(vocab=vocab, pad_value=0., seq_length=max_seq_length)
    x_test, y_test, test_xs = preprocessor.transform_data(seqs_test, xs=xs_test)
    print "(test sequences,test timesteps,input dimension):", x_test.shape

    if with_xs:
        return model.model.evaluate([x_test, test_xs], y_test)
    else:
        return model.model.evaluate(x_test, y_test)


def analyze_history(history):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    print "min val loss: %f at epoch: %d" % (np.min(val_losses), np.argmin(val_losses) + 1)
    print "train loss: %f at epoch: %d" % (train_losses[np.argmin(val_losses)], np.argmin(val_losses) + 1)
    results = ModelResults(train_losses[np.argmin(val_losses)], np.min(val_losses), np.argmin(val_losses) + 1)
    return results


def run_multinomial(seqs_train, val_seqs, n_classes, wrt_time=False, normalize=True,k=1.0):
    model = MultinomialModel(n_classes, model_name="multinomial", k=k)
    model.fit_model(seqs_train, normalize=normalize)
    val_preds = model.predict(val_seqs)
    if wrt_time:
        train_neg_ll, val_neg_ll = utils.compute_likelihood_cut(val_preds, 0.7, count_first_prob=False)
    else:
        train_preds = model.predict(seqs_train)
        train_neg_ll = utils.compute_likelihood(train_preds, count_first_prob=False)
        val_neg_ll = utils.compute_likelihood(val_preds, count_first_prob=False)
    print "Multinomial train neg ll: %f, val neg ll: %f" % (train_neg_ll, val_neg_ll)
    results = ModelResults(train_neg_ll, val_neg_ll, None)
    return model, results


def run_markov(seqs_train, val_seqs, n_classes, wrt_time=False, k=1.0):
    model = MarkovModel(n_classes, model_name="markov", order=1, k=k)
    model.fit_model(seqs_train)
    val_preds = model.predict(val_seqs)
    if wrt_time:
        train_neg_ll, val_neg_ll = utils.compute_likelihood_cut(val_preds, 0.7, count_first_prob=False)
    else:
        train_preds = model.predict(seqs_train)
        train_neg_ll = utils.compute_likelihood(train_preds, count_first_prob=False)
        val_neg_ll = utils.compute_likelihood(val_preds, count_first_prob=False)
    print "Markov train neg ll: %f, val neg ll: %f" % (train_neg_ll, val_neg_ll)
    results = ModelResults(train_neg_ll, val_neg_ll, None)
    return model, results


def run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,orig_seqs_lengths=None, wrt_time=False,
                            read_file=None, model_checkpoint=False, early_stopping=False, n_epochs=50, batch_size=10,
                            y_to_y_trainable=True, y_to_y_w_initializer=None, y_to_y_regularizer=None,
                            verbose=1, model_name="test_model", connect_x=True, connect_y=True,
                            xy_bias=False, y_bias=False, embed_y=False, z_dim=10, diag_b=True):
    model = NoRecurrenceModel(timesteps=x_train.shape[1],x_dim=x_train.shape[2], y_dim=len(vocab),
                              model_name=model_name,
                              y_to_y_activation="linear", y_to_y_w_initializer=y_to_y_w_initializer,
                              x_to_y_activation="linear", out_activation="softmax",
                              y_bias=y_bias, xy_bias=xy_bias, connect_x=connect_x,
                              y_to_y_regularizer=y_to_y_regularizer,
                              connect_y=connect_y, embed_y=embed_y, z_dim=z_dim, diag_b=diag_b)
    if connect_x:
        if connect_y:
            train = [x_train, train_xs]
            validation = [x_val, val_xs]
        else:
            train = train_xs
            validation = val_xs
    else:
        train = x_train
        validation = x_val

    if read_file:
        model.load_model_weights(read_file)
        model.compile_model(loss='categorical_crossentropy', metrics=[], optimizer='adam')
        results = ModelResults()
        metrics_names, scores = model.evaluate(train, y_train, batch_size=1)
        results.train_loss = scores[0]
        metrics_names, scores = model.evaluate(validation, y_val, batch_size=1)
        results.val_loss = scores[0]
        print "train loss: %f, val loss: %f" % (results.train_loss, results.val_loss)
    else:
        if connect_y:
            model.set_layer_weights_trainable("y_output", trainable=y_to_y_trainable)
        history = run_model(model, train, y_train, validation_data=(validation, y_val),
                            orig_seqs_lengths=orig_seqs_lengths,model_checkpoint=model_checkpoint,
                            n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                            early_stopping=early_stopping, wrt_time=wrt_time)
        results = analyze_history(history, model.model_name)
    return model, results


def run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs,
                              vocab,orig_seqs_lengths=None,
                              wrt_time=False, y_to_y_trainable=True, y_to_y_w_initializer=None, toy_regularizer=None,
                              y_to_y_regularizer=None, read_file=None, rnn_type="LSTM", early_stopping=False,
                              n_epochs=50,model_checkpoint=True,
                              batch_size=10, verbose=1, z_dim=10, model_name="test_model", y_to_z=True, y_to_y=True,
                              x_to_y=True, x_to_z=False, diag_b=True, z_to_y_drop=0.0, y_to_z_dropout=0.1):
    model = RNNFullModel(timesteps=x_train.shape[1], x_dim=x_train.shape[2], y_dim=len(vocab), z_dim=z_dim,
                         model_name=model_name, rnn_type=rnn_type, z_to_z_activation="relu",
                         y_to_y_activation="linear", ytoy_bias=False, z_bias=True, toy_bias=False,
                         toy_regularizer=toy_regularizer,
                         y_to_y_regularizer=y_to_y_regularizer, y_to_y_w_initializer=y_to_y_w_initializer,
                         y_to_z=y_to_z, y_to_y=y_to_y, x_to_y=x_to_y, x_to_z=x_to_z, diag_b=diag_b,
                         z_to_y_drop=z_to_y_drop, y_to_z_dropout=y_to_z_dropout)
    train = []
    validation = []
    if y_to_y or y_to_z:
        train.append(x_train)
        validation.append(x_val)
    if x_to_y or x_to_z:
        train.append(train_xs)
        validation.append(val_xs)
    if read_file:
        model.load_model_weights(read_file)
        model.compile_model(loss='categorical_crossentropy', metrics=[], optimizer='adam')
        results = ModelResults()
        metrics_names, scores = model.evaluate(train, y_train)
        results.train_loss = scores[0]
        metrics_names, scores = model.evaluate(validation, y_val)
        results.val_loss = scores[0]
        print results.train_loss, results.val_loss
        print "train loss: %f, val loss: %f" % (results.train_loss, results.val_loss)
    else:
        if y_to_y:
            model.set_layer_weights_trainable("y_to_y_output", trainable=y_to_y_trainable)
            trainable_weights, non_trainable_weights = model.get_model_weights()
            # print "trainable weights: %s, non trainable weights %s" % (trainable_weights, non_trainable_weights)
        # print model.model_name
        # print model.model.summary()
        model.compile_model(loss='categorical_crossentropy', metrics=[], optimizer='adam')

        history = run_model(model, train, y_train, validation_data=(validation, y_val),
                            orig_seqs_lengths=orig_seqs_lengths,model_checkpoint=model_checkpoint,
                            n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                            early_stopping=early_stopping, wrt_time=wrt_time)
        results = analyze_history(history, model.model_name)

    return model, results
