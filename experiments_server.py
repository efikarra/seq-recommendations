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

def run_k_fold(seqs,splits,dataset_name):
    split_wrt_time=False
    max_seq_length = utils.compute_seq_max_length(seqs) - 1
    for i in range(splits):
        with open('splits/'+dataset_name+'-split%d'%i+'.pickle', 'rb') as handle:
            (train_idxs,val_idxs)=pickle.load(handle)
        print "fold %d" % i
        train_seqs = [seqs[i] for i in train_idxs]
        print ""
        val_seqs = [seqs[i] for i in val_idxs]

        n_epochs = 600
        small_n_epochs=300
        rnn_type = "simpleRNN"
        z_dim=10
        batch_size=100
        k = 10e-7

        xs_train = datasets.build_xs(train_seqs, vocab, freq=False)
        xs_val = datasets.build_xs(val_seqs, vocab, freq=False)

        print "vocabulary size:", len(vocab)
        print "train sequences:", len(train_seqs)
        print "val sequences:", len(val_seqs)
        print
        print "multinomial"
        multinomial, results = run_multinomial(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time,k=k)

        print
        print "markov"
        markov, results = run_markov(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time, k=k)

        train_freqs, gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=True, end_state=False)
        # build train/val sets for rnns
        x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_model_input(train_seqs, val_seqs, xs_train,
                                                                               xs_val, vocab,
                                                                               max_seq_length=max_seq_length)

        var = 1.0
        init_weights = utils.sample_weights(np.log(train_freqs), np.sqrt(var))
        print
        print "ytoy1"
        model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                 wrt_time=split_wrt_time, early_stopping=False,
                                                 model_checkpoint=True,n_epochs=small_n_epochs,
                                                 batch_size=batch_size, y_to_y_regularizer=gauss_prior(np.log(train_freqs),var),
                                                 y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                 verbose=0, model_name="ytoy1", y_bias=False,
                                                 connect_x=False, connect_y=True)

        print
        print "ytoy_xtoy"
        model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                 model_checkpoint=True, orig_seqs_lengths=None, wrt_time=split_wrt_time,
                                                 y_to_y_regularizer=gauss_prior(np.log(train_freqs), var),
                                                 y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                 early_stopping=False, n_epochs=small_n_epochs, batch_size=batch_size,
                                                 verbose=2, model_name="ytoy_xtoy1", connect_x=True, y_bias=False,
                                                 connect_y=True, diag_b=True, read_file=None)


        var = 0.01
        init_weights = utils.sample_weights(np.log(train_freqs), np.sqrt(var))
        print
        print "ytoy2"
        model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                 wrt_time=split_wrt_time, early_stopping=False, n_epochs=n_epochs,
                                                 batch_size=batch_size,model_checkpoint=True,
                                                 y_to_y_regularizer=None,
                                                 y_to_y_w_initializer=None,
                                                 verbose=2, model_name="ytoy2", y_bias=False,
                                                 connect_x=False, connect_y=True)

        print
        print "ytoy_xtoy2"
        model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                 model_checkpoint=True, orig_seqs_lengths=None, wrt_time=split_wrt_time,
                                                 y_to_y_regularizer=gauss_prior(np.log(train_freqs), var),
                                                 y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                 early_stopping=False, n_epochs=small_n_epochs, batch_size=batch_size,
                                                 verbose=2, model_name="ytoy_xtoy2", connect_x=True, y_bias=False,
                                                 connect_y=True, diag_b=True, read_file=None)

        print
        print "ytoz"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   toy_regularizer=None,
                                                   orig_seqs_lengths=None, wrt_time=False, read_file=None,
                                                   rnn_type=rnn_type, early_stopping=False,model_checkpoint=True,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                                   model_name="ytoz", y_to_z=True, y_to_y=False,
                                                   x_to_y=False, x_to_z=False, z_to_y_drop=0.1)

        print
        print "ytoz_ytoy1"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_regularizer=gauss_prior(np.log(train_freqs),var),
                                                   model_checkpoint=True,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   read_file=None, rnn_type=rnn_type, early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                                   model_name="ytoz_ytoy1", y_to_z=True, y_to_y=True,
                                                   x_to_y=False, x_to_z=False, z_to_y_drop=0.1)

        var = 1.0
        init_weights = utils.sample_weights(np.log(train_freqs), np.sqrt(var))
        print
        print "ytoz_ytoy2"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_regularizer=gauss_prior(np.log(train_freqs), var),
                                                   model_checkpoint=True,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   read_file=None, rnn_type=rnn_type, early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=z_dim,
                                                   model_name="ytoz_ytoy2", y_to_z=True, y_to_y=True,
                                                   x_to_y=False, x_to_z=False, z_to_y_drop=0.1)
        print
        print "ytoz_xtoz"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_regularizer=gauss_prior(np.log(train_freqs),var), read_file=None,
                                                   rnn_type=rnn_type,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=0, z_dim=z_dim,
                                                   model_name="ytoz_xtoz", y_to_z=True, y_to_y=False,
                                                   x_to_y=False, x_to_z=True, z_to_y_drop=0.1)

        print
        print "ytoz_ytoy_xtoz"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_regularizer=gauss_prior(np.log(train_freqs),var), read_file=None,
                                                   rnn_type=rnn_type,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=0, z_dim=z_dim,
                                                   model_name="ytoz_ytoy_xtoz", y_to_z=True, y_to_y=True,
                                                   x_to_y=False, x_to_z=True, z_to_y_drop=0.1)

        print
        print "ytoz_ytoy_xtoy"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_regularizer=gauss_prior(np.log(train_freqs),var),
                                                   read_file=None,
                                                   rnn_type=rnn_type,model_checkpoint=True,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=2, z_dim=10,
                                                   model_name="ytoz_ytoy_xtoy", y_to_z=True, y_to_y=True,
                                                   x_to_y=True, x_to_z=False, z_to_y_drop=0.1)


if __name__ == "__main__":
    # load flickr data
    print "Experimenting with Flickr:"
    seqs, vocab = datasets.load_flickr_data()
    seqs = datasets.remove_short_seqs(seqs, min_seq_length=2)
    print "total seqs,vocab: ", len(seqs), len(vocab)
    run_k_fold(seqs, 1, dataset_name="msnbc")
    print "Experimenting with Flickr ended."

    print "Experimenting with MSNBC:"
    seqs, vocab = datasets.load_msnbc_data()
    print "total vocab: ", len(vocab)
    import random
    # run experiments with MSNBC real and simulated data
    seqs = datasets.remove_short_seqs(seqs, min_seq_length=2)
    with open('splits/msnbc-sample.pickle', 'rb') as handle:
        dataset_sample=pickle.load(handle)

    seqs = [seqs[i] for i in dataset_sample]
    run_k_fold(seqs, 1, dataset_name="msnbc")
    print "Experimenting with MSNBC ended."
