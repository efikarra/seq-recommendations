"""
Main method for performing experiments
"""
from experiments_methods import *
import datasets
from model import gauss_prior, ArrayInitializer
from keras.regularizers import l2
from keras.initializers import Constant
from sklearn.model_selection import ShuffleSplit
from collections import namedtuple
import pickle
import itertools


def run_experiments(dataset_name):
    print "dataset ", dataset_name
    with open('paper_sets/'+dataset_name+'_train.pickle', 'rb') as handle:
        train_seqs=pickle.load(handle)
    with open('paper_sets/'+dataset_name+'_val.pickle', 'rb') as handle:
        val_seqs=pickle.load(handle)
    with open('paper_sets/' + dataset_name +'_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    print "vocabulary size:", len(vocab)
    print "train sequences:", len(train_seqs)
    print "val sequences:", len(val_seqs)
    print

    max_se_q_length1 = utils.compute_seq_max_length(train_seqs) - 1
    max_se_q_length2 = utils.compute_seq_max_length(val_seqs) - 1
    max_se_q_length=max(max_se_q_length1,max_se_q_length2)

    xs_train = datasets.build_xs(train_seqs, vocab, freq=True)
    xs_val = datasets.build_xs(val_seqs, vocab, freq=True)
    xs_train = [[[np.log(x + 1) for x in xs] for xs in xss] for xss in xs_train]
    xs_val = [[[np.log(x + 1) for x in xs] for xs in xss] for xss in xs_val]

    n_epochs = 500
    small_n_epochs=200
    rnn_type = "LSTM"
    z_dim=100
    batch_size=100
    k = 10e-7


    print "multinomial"
    multinomial, results = run_multinomial(train_seqs, val_seqs, len(vocab), wrt_time=False,k=k)

    print
    print "markov"
    markov, results = run_markov(train_seqs, val_seqs, len(vocab), wrt_time=False, k=k)

    train_freqs, gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=True, end_state=False)
    # # build train/val sets for rnns
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_model_input(train_seqs, val_seqs, xs_train,
                                                                           xs_val, vocab,
                                                                           max_seq_length=max_se_q_length)

    var = 0.01
    init_weights = utils.sample_weights(np.log(train_freqs), np.sqrt(var))
    print
    init_weights = np.log(train_freqs)
    print "ytoy_reg"
    model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                             wrt_time=False, early_stopping=True,
                                             model_checkpoint=True,n_epochs=small_n_epochs,y_to_y_trainable=False,
                                             batch_size=batch_size, y_to_y_regularizer=None, read_file=None,
                                             y_to_y_w_initializer=ArrayInitializer(init_weights),
                                             verbose=2, model_name=dataset_name+"_ytoy_reg", y_bias=False,
                                             connect_x=False, connect_y=True)

    print
    print "ytoy"
    model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                             wrt_time=False, early_stopping=False,
                                             model_checkpoint=True, n_epochs=n_epochs,
                                             batch_size=batch_size,y_to_y_trainable=True,
                                             y_to_y_regularizer=None,
                                             y_to_y_w_initializer=None, read_file=None,
                                             verbose=2, model_name=dataset_name+"_ytoy", y_bias=False,
                                             connect_x=False, connect_y=True)


    print
    print "ytoy_xtoy_fixed"
    model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                             model_checkpoint=True, orig_seqs_lengths=None, wrt_time=False,
                                             y_to_y_regularizer=None,
                                             y_to_y_w_initializer=ArrayInitializer(init_weights),y_to_y_trainable=False,
                                             early_stopping=True, n_epochs=n_epochs, batch_size=batch_size,
                                             verbose=2, model_name=dataset_name+"_ytoy_xtoy_fixed", connect_x=True, y_bias=False,
                                             connect_y=True, diag_b=True, read_file=None)

    print
    print "ytoy_xtoy"
    model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                             model_checkpoint=True, orig_seqs_lengths=None, wrt_time=False,
                                             y_to_y_regularizer=None,
                                             y_to_y_w_initializer=None,
                                             early_stopping=False, n_epochs=n_epochs, batch_size=batch_size,
                                             verbose=2, model_name=dataset_name+"_ytoy_xtoy", connect_x=True, y_bias=False,
                                             connect_y=True, diag_b=True, read_file=None)



    print
    print "ytoz"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               toy_regularizer=None,
                                               orig_seqs_lengths=None, wrt_time=False, read_file=None,
                                               rnn_type=rnn_type, early_stopping=False,model_checkpoint=True,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name+"_ytoz", y_to_z=True, y_to_y=False,
                                               x_to_y=False, x_to_z=False, z_to_y_drop=0.3)

    print
    print "ytoz_ytoy_fixed"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=False,
                                               y_to_y_regularizer=None,
                                               model_checkpoint=True,
                                               y_to_y_w_initializer=ArrayInitializer(init_weights),
                                               read_file=None, rnn_type=rnn_type, early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name+"_ytoz_ytoy_fixed", y_to_z=True, y_to_y=True,
                                               x_to_y=False, x_to_z=False, z_to_y_drop=0.3)

    print
    print "ytoz_ytoy"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                               y_to_y_regularizer=None,
                                               model_checkpoint=True,
                                               y_to_y_w_initializer=None,
                                               read_file=None, rnn_type=rnn_type, early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name+"_ytoz_ytoy", y_to_z=True, y_to_y=True,
                                               x_to_y=False, x_to_z=False, z_to_y_drop=0.3)



    print
    print "ytoz_xtoz"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                               y_to_y_regularizer=None, read_file=None,
                                               rnn_type=rnn_type, model_checkpoint=True,
                                               y_to_y_w_initializer=None,
                                               early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=1, z_dim=z_dim,
                                               model_name=dataset_name+"ytoz_xtoz", y_to_z=True, y_to_y=False,
                                               x_to_y=False, x_to_z=True, z_to_y_drop=0.3)
    #
    print
    print "ytoz_ytoy_xtoz"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=False,
                                               y_to_y_regularizer=None, read_file=None,
                                               rnn_type=rnn_type, model_checkpoint=True,
                                               y_to_y_w_initializer=ArrayInitializer(init_weights),
                                               early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name+"ytoz_ytoy_xtoz", y_to_z=True, y_to_y=True,
                                               x_to_y=False, x_to_z=True, z_to_y_drop=0.3)
    #
    print
    print "_ytoz_ytoy_xtoy_fixed"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=False,
                                               y_to_y_regularizer=None,
                                               read_file=None,
                                               rnn_type=rnn_type,model_checkpoint=True,
                                               y_to_y_w_initializer=ArrayInitializer(init_weights),
                                               early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name+"_ytoz_ytoy_xtoy_fixed", y_to_z=True, y_to_y=True,
                                               x_to_y=True, x_to_z=False, z_to_y_drop=0.3)

    print
    print "_ytoz_ytoy_xtoy"
    model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                               orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                               y_to_y_regularizer=None,
                                               read_file=None,
                                               rnn_type=rnn_type, model_checkpoint=True,
                                               y_to_y_w_initializer=None,
                                               early_stopping=False,
                                               n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                               model_name=dataset_name + "_ytoz_ytoy_xtoy", y_to_z=True,
                                               y_to_y=True,
                                               x_to_y=True, x_to_z=False, z_to_y_drop=0.3)


if __name__ == "__main__":
    # run_experiments(dataset_name="flickr")
    run_experiments(dataset_name="msnbc")
    # run_experiments(dataset_name="reddit")
    # run_experiments(dataset_name="gowalla")
    # run_experiments(dataset_name="reddit_elim")
    # run_experiments(dataset_name="reddit")
    # run_experiments(dataset_name="student")
    # run_experiments(dataset_name="switch")
