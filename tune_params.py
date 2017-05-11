"""
Main method for performing experiments
"""
from experiments_methods import *
import datasets
from model import gauss_prior, ArrayInitializer
from keras.regularizers import l2
from keras.initializers import Constant,RandomNormal,RandomUniform
from sklearn.model_selection import ShuffleSplit
from collections import namedtuple
import pickle
import itertools
import sys
import json
import time

def chop_sequences(seqs, offset=200):
    chooped_seqs = []
    for seq in seqs:
        chooped_seqs.append(seq[offset:])
    return chooped_seqs


def run_experiments(dataset_name,filename,model_name="ytoz"):
    print
    print "dataset ", dataset_name
    with open('paper_sets/' + dataset_name + '_train.pickle', 'rb') as handle:
        train_seqs = pickle.load(handle)
    with open('paper_sets/' + dataset_name + '_val.pickle', 'rb') as handle:
        val_seqs = pickle.load(handle)
    with open('paper_sets/' + dataset_name + '_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    print "vocabulary size:", len(vocab)
    print "train sequences:", len(train_seqs)
    print "val sequences:", len(val_seqs)
    print
    k = 10e-7
    max_se_q_length1 = utils.compute_seq_max_length(train_seqs) - 1
    max_se_q_length2 = utils.compute_seq_max_length(val_seqs) - 1
    max_se_q_length = max(max_se_q_length1, max_se_q_length2)

    xs_train = datasets.build_xs(train_seqs, vocab, freq=True)
    xs_val = datasets.build_xs(val_seqs, vocab, freq=True)
    xs_train = [[[np.log(x + 1) for x in xs] for xs in xss] for xss in xs_train]
    xs_val = [[[np.log(x + 1) for x in xs] for xs in xss] for xss in xs_val]
    # train_freqs, gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=True, end_state=False)
    # init_weights = np.log(train_freqs)
    # # build train/val sets for rnns
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_model_input(train_seqs, val_seqs, xs_train,
                                                                           xs_val, vocab,
                                                                           max_seq_length=max_se_q_length)
    train_freqs, gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=True, end_state=False)
    init_weights = np.log(train_freqs)

    start = time.time()
    n_epochs = 1500
    batch_size = 100
    # params = {"z_dim": [500,1000],
    #            "z_to_z_dropout": [0.25,0.5,0.75],
    #           "rnn_type": ["LSTM"],"lr":[0.1,0.01,0.001,0.0001]}
    params=json.load(open(filename))
    all_names = sorted(params)
    params_combinations = list(itertools.product(*(params[name] for name in all_names)))
    vall_losses={}
    train_losses = {}
    epochs = {}
    print "Number of combinations: ", len(params_combinations)
    for i,combination in enumerate(params_combinations):
        print
        print "combination %d: "%i, combination
        comb_name = "lr_" + str(combination[0]) + "_" + str(combination[1]) + "_z_" + str(combination[2]) + \
                    "_zz_" + str(combination[3])

        if model_name=="ytoz":
            print "ytoz"
            print comb_name
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                       toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                       model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                       z_dim=combination[2],y_to_z_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None),
                                                       model_name=dataset_name + "ytoz"+comb_name, y_to_z=True, y_to_y=False,
                                                       x_to_y=False, x_to_z=False, z_to_y_drop=combination[3],
                                                       y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                       lr=combination[0])
        if model_name == "ytoz_xtoz":
            print
            print "ytoz_xtoz"
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                       toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                       model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                       z_dim=combination[2],y_to_z_initializer=RandomNormal(minval=-0.01, maxval=0.01, seed=None),
                                                       model_name=dataset_name + "ytoz_xtoz"+comb_name, y_to_z=True, y_to_y=False,
                                                       x_to_y=False, x_to_z=True, z_to_y_drop=combination[3],
                                                       y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                       lr=combination[0])
        if model_name == "ytoz_ytoy_xtoz_fixed":
            print
            print "ytoz_ytoy_xtoz_fixed"
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                           toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                           model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                           z_dim=combination[2],y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                           y_to_y_trainable=False,
                                                           model_name=dataset_name + "ytoz_ytoy_xtoz_fixed"+comb_name, y_to_z=True, y_to_y=True,
                                                           x_to_y=False, x_to_z=True, z_to_y_drop=combination[3],
                                                           y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                           lr=combination[0])


        if model_name == "ytoz_ytoy_xtoz":
            print
            print "ytoz_ytoy_xtoz"
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                           toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                           model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                           z_dim=combination[2],y_to_y_w_initializer=None,
                                                           y_to_y_trainable=True,
                                                           model_name=dataset_name + "ytoz_ytoy_xtoz"+comb_name, y_to_z=True, y_to_y=True,
                                                           x_to_y=False, x_to_z=True, z_to_y_drop=combination[3],
                                                           y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                           lr=combination[0])

        if model_name == "ytoz_ytoy_xtoy_fixed":
            print
            print "ytoz_ytoy_xtoy_fixed"
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                       toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                       model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                       z_dim=combination[2],y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                       y_to_y_trainable=False,
                                                       model_name=dataset_name + "ytoz_ytoy_xtoy_fixed"+comb_name, y_to_z=True, y_to_y=True,
                                                       x_to_y=True, x_to_z=False, z_to_y_drop=combination[3],
                                                       y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                       lr=combination[0])

        if model_name == "ytoz_ytoy_xtoy":
            print
            print "ytoz_ytoy_xtoy_fixed"
            model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                       toy_regularizer=None,rnn_type=combination[1], early_stopping=True,
                                                       model_checkpoint=True,n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                       z_dim=combination[2],y_to_y_w_initializer=None,
                                                       y_to_y_trainable=True,
                                                       model_name=dataset_name + "ytoz_ytoy_xtoy"+comb_name, y_to_z=True, y_to_y=True,
                                                       x_to_y=True, x_to_z=False, z_to_y_drop=combination[3],
                                                       y_to_z_dropout=0.0,z_to_z_dropout=combination[3],
                                                       lr=combination[0])
        vall_losses[combination]=results.val_loss
        train_losses[combination]=results.train_loss
        epochs[combination]=results.epoch

    print "all val losses: ",vall_losses
    print "all train losses: ",train_losses
    print "all epochs: ",epochs

    min_val=np.argmin(vall_losses.values)
    best_comb=params_combinations[min_val]
    print "best combination: ",best_comb
    print "best loss: ",np.min(vall_losses[best_comb])
    end = time.time()
    print "time:", end - start


if __name__ == "__main__":
    # run_experiments(dataset_name="flickr", filename="file-params1.txt", model_name="ytoz")
    run_experiments(dataset_name=sys.argv[1],filename=sys.argv[2],model_name=sys.argv[3])

