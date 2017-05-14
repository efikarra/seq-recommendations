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


def chop_sequences(seqs, offset=200):
    chooped_seqs = []
    for seq in seqs:
        chooped_seqs.append(seq[offset:])
    return chooped_seqs


def run_experiments(dataset_name):
    print "dataset ", dataset_name
    with open('train_sets/' + dataset_name + '_train.pickle', 'rb') as handle:
        train_seqs = pickle.load(handle)
    with open('val_sets/' + dataset_name + '_val.pickle', 'rb') as handle:
        val_seqs = pickle.load(handle)
    with open('vocabs/' + dataset_name + '_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    print "vocabulary size:", len(vocab)
    print "train sequences:", len(train_seqs)
    print "val sequences:", len(val_seqs)
    print
    k = 10e-7
    max_se_q_length1 = utils.compute_seq_max_length(train_seqs) - 1
    max_se_q_length2 = utils.compute_seq_max_length(val_seqs) - 1
    max_se_q_length = max(max_se_q_length1, max_se_q_length2)

    xs_train = datasets.build_xs(train_seqs, vocab, freq=False)
    xs_val = datasets.build_xs(val_seqs, vocab, freq=False)

    train_freqs, gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=True, end_state=False)
    # # build train/val sets for rnns
    x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_model_input(train_seqs, val_seqs, xs_train,
                                                                           xs_val, vocab,
                                                                           max_seq_length=max_se_q_length)

    n_epochs = 1
    batch_size = 10
    init_weights = np.log(train_freqs)

    params = {"z_dim": [100,200,500,1000], "z_to_y_drop": [0.2,0.3,0.4,0.5],
              "y_to_z_dropout": [0.2,0.3,0.4,0.5], "z_to_z_dropout": [0.2,0.3,0.4,0.5],
              "rnn_type": ["simpleRNN","LSTM"]}

    all_names = sorted(params)
    params_combinations = list(itertools.product(*(params[name] for name in all_names)))
    vall_losses={}
    train_losses = {}
    epochs = {}
    print "Number of combinations: ", len(params_combinations)
    for i,combination in enumerate(params_combinations):
        if i>=3: break
        print
        print "combination %d: "%i, combination
        print "ytoz"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   toy_regularizer=None,
                                                   orig_seqs_lengths=None, wrt_time=False, read_file=None,
                                                   rnn_type=combination[0], early_stopping=False, model_checkpoint=True,
                                                   n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                                                   z_dim=combination[2],
                                                   model_name=dataset_name + "_ytoz%d"%i, y_to_z=True, y_to_y=False,
                                                   x_to_y=False, x_to_z=False, z_to_y_drop=combination[3],
                                                   y_to_z_dropout=combination[1],
                                                   z_to_z_dropout=combination[4])
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



if __name__ == "__main__":
    # run_experiments(dataset_name="flickr")
    # run_experiments(dataset_name="msnbc")
    # run_experiments(dataset_name="reddit")
    # run_experiments(dataset_name="gowalla")
    # run_experiments(dataset_name="reddit_elim")
    # run_experiments(dataset_name="reddit")
    run_experiments(dataset_name="student")
    # run_experiments(dataset_name="switch")
