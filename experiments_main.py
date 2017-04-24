"""
Main method for performing experiments
"""
from experiments_methods import *
from sampler import MCSampler
import datasets
from model import gauss_prior


def run_experiments(seqs, vocab):
    max_seq_length=utils.compute_seq_max_length(seqs)
    y_to_y_regularizer = None
    # run all models. test seqs not used now. we only split and compute train and val performance
    # run all models splitting w.r.t time into train/val sets.
    split_wrt_time = False
    # original seq lengths are needed only in case we split w.r.t time, for RNNs to compute my custom val loss function at each
    # timestep. Also here, we run models with val_seqs=seqs since we compute the likelihhod for the whole sequences and
    # for validation, we count only the validation last part of the each seq.

    if split_wrt_time:
        # convert DataFrame into list of lists and compute x's and vocabulary.
        orig_seqs_lengths = [len(seq) - 1 for seq in seqs]
        xs = datasets.build_xs(seqs, vocab)
        train_seqs, val_seqs, seqs_test, xs_train, xs_val, xs_test = datasets.split_seqs_wrt_time(seqs, xs, train=0.8,
                                                                                                  val=0.2, test=0.0)
        alpha, gamma = utils.transition_matrix(train_seqs, len(vocab), k=1.0, freq=False, end_state=False)
        y_to_y_regularizer = gauss_prior(alpha, 1)
        y_to_y_regularizer = None

        run_multinomial(train_seqs, seqs, len(vocab), wrt_time=split_wrt_time)
        run_markov(train_seqs, seqs, len(vocab), wrt_time=split_wrt_time)
        run_baseline_rnn(train_seqs, seqs, xs_train, xs, vocab, orig_seqs_lengths=orig_seqs_lengths, with_xs=True,
                         wrt_time=split_wrt_time, rnn_type="LSTM", early_stopping=False, n_epochs=50)
        run_rnn_xtoy(train_seqs, seqs, xs_train, xs, vocab, orig_seqs_lengths=orig_seqs_lengths,
                     wrt_time=split_wrt_time,
                     rnn_type="LSTM", early_stopping=False, n_epochs=50)
        run_rnn_ytoy(train_seqs, seqs, xs_train, xs, vocab, orig_seqs_lengths=orig_seqs_lengths, with_xs=True,
                     wrt_time=split_wrt_time, rnn_type="LSTM", early_stopping=False, n_epochs=50, y_to_y_trainable=True,
                     y_to_y_regularizer=y_to_y_regularizer)
        run_rnn_fullmodel(train_seqs, seqs, xs_train, xs, vocab, orig_seqs_lengths=orig_seqs_lengths,
                          wrt_time=split_wrt_time,
                          rnn_type="LSTM", early_stopping=False, n_epochs=50, y_to_y_trainable=True,
                          y_to_y_regularizer=y_to_y_regularizer)
    else:
        # run all models splitting normally into train/val sets, i.e., use completely new sequences for validation.
        # We split the DataFrames first, i.e the original data and then build sequences based on vocabulary because we have to compute the vocab only from train set.
        # Rarely, since train/val splitting is random, 1 class exists on val set but not on train set. In that case the entries of that class are removed from the val set.
        orig_seqs_lengths = None
        train_seqs, val_seqs, test_seqs = datasets.split_seqs(seqs, shuffle=False, train=0.7, val=0.3, test=0.0)
        xs_train = datasets.build_xs(train_seqs, vocab, freq=False)
        xs_val = datasets.build_xs(val_seqs, vocab, freq=False)

        print "vocabulary size:", len(vocab)
        print "train sequences:", len(train_seqs)
        print "val sequences:", len(val_seqs)
        run_multinomial(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time)
        run_markov(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time)

        # run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=False,
        #                  wrt_time=split_wrt_time, rnn_type="simpleRNN", z_dim=10, z_activation="tanh",
        #                  early_stopping=False, n_epochs=50, batch_size=10, verbose=0,
        #                  model_name="baseline_noxs_rnn")
        # run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=True,
        #                  wrt_time=split_wrt_time, rnn_type="simpleRNN", z_dim=10, early_stopping=False, n_epochs=50,
        #                  batch_size=10, verbose=0,
        #                  model_name="baseline_xs_rnn")
        #
        # run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=False,
        #                  wrt_time=split_wrt_time, rnn_type="LSTM", z_dim=10, z_activation="tanh", early_stopping=False,
        #                  n_epochs=50, batch_size=10, verbose=0,
        #                  model_name="baseline_noxs_lstm")
        # run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=True,
        #                  wrt_time=split_wrt_time, rnn_type="LSTM", z_dim=10, early_stopping=False, n_epochs=50,
        #                  batch_size=10, verbose=0,
        #                  model_name="baseline_xs_lstm")

        run_rnn_only_ytoy(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None,
                          wrt_time=split_wrt_time,
                          early_stopping=False, n_epochs=80, batch_size=10,
                          verbose=0, model_name="only_y_to_y_noxs", connect_x=False, y_bias=False,
                          connect_y=True, embed_y=False, z_dim=10)

        run_rnn_only_ytoy(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None,
                                     wrt_time=split_wrt_time,
                                     early_stopping=False, n_epochs=80, batch_size=10,
                                     verbose=0, model_name="only_y_to_y_withxs", connect_x=True, y_bias=False,
                                     connect_y=True, embed_y=False, z_dim=10)

        # run_rnn_fullmodel(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=orig_seqs_lengths,
        #                   wrt_time=split_wrt_time, rnn_type="simpleRNN", early_stopping=False, n_epochs=50,
        #                   y_to_y_trainable=True, verbose=0, y_to_y_regularizer=None, model_name="fullmodel_rnn", z_dim=10)
        #
        # run_rnn_fullmodel(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=orig_seqs_lengths,
        #                   wrt_time=split_wrt_time, rnn_type="LSTM", early_stopping=False, n_epochs=50, y_to_y_trainable=True,
        #                   verbose=0, y_to_y_regularizer=None, model_name="fullmodel_lstm", z_dim=10)
        #
        # weights=utils.transition_matrix(train_seqs, len(vocab), k=1.0, freq=False, end_state=False)
        # run_rnn_fullmodel(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=orig_seqs_lengths,
        #                   wrt_time=split_wrt_time, rnn_type="simpleRNN", early_stopping=False, n_epochs=50,
        #                   y_to_y_trainable=False, verbose=0, y_to_y_regularizer=None, y_to_y_weights=np.log(weights),
        #                   model_name="fullmodel_rnn_init_a", z_dim=10)

        # run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, orig_seqs_lengths=orig_seqs_lengths,
        #                  with_xs=True, wrt_time=split_wrt_time, rnn_type="simpleRNN", early_stopping=False, batch_size=50, n_epochs=50,
        #                  verbose=1, model_name="baseline_with_xs")
        # run_rnn_xtoy(train_seqs, val_seqs, xs_train, xs_val, vocab, orig_seqs_lengths=orig_seqs_lengths, wrt_time=split_wrt_time,
        #               rnn_type="LSTM", early_stopping=False, n_epochs=50)
        # run_rnn_ytoy(seqs_train, seqs_val, xs_train, xs_val, vocab, orig_seqs_lengths=orig_seqs_lengths, wrt_time=split_wrt_time,
        #              rnn_type="LSTM", early_stopping=False, n_epochs=50, y_to_y_trainable=True, y_to_y_regularizer=y_to_y_regularizer)
        # run_rnn_fullmodel(seqs_train, seqs_val, xs_train, xs_val, vocab, orig_seqs_lengths=orig_seqs_lengths,
        #                   wrt_time=split_wrt_time, rnn_type="LSTM", early_stopping=False, n_epochs=50, y_to_y_trainable=True, y_to_y_regularizer=y_to_y_regularizer)


def run_experiments_on_simulated_data(seqs, vocab, n_seqs=1000):
    alpha, gamma = utils.transition_matrix(seqs, len(vocab), k=1.0, freq=False, end_state=False)
    sampler = MCSampler(alpha, gamma, beta=0.1, use_end_token=False)
    seqs = []
    for _ in xrange(n_seqs):
        seqs.append(sampler.gen_sequence(n=40))
    max_seq_length = utils.compute_seq_max_length(seqs)
    split_wrt_time = False
    train_seqs, val_seqs, test_seqs = datasets.split_seqs(seqs, shuffle=False, train=0.8, val=0.2, test=0.0)
    xs_train = datasets.build_xs(train_seqs, vocab, freq=False)
    xs_val = datasets.build_xs(val_seqs, vocab, freq=False)
    print "total train sequences:", len(train_seqs)
    print "total val sequences:", len(val_seqs)
    print "total val sequences:", len(test_seqs)
    run_multinomial(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time)
    run_markov(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time)
    run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=False,
                     wrt_time=split_wrt_time, rnn_type="simpleRNN", z_dim=10, z_activation="tanh", early_stopping=False, n_epochs=50, batch_size=10, verbose=1,
                     model_name="baseline_noxs_rnn")
    run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=True,
                     wrt_time=split_wrt_time, rnn_type="simpleRNN", z_dim=10, early_stopping=False, n_epochs=50,
                     batch_size=10, verbose=1,
                     model_name="baseline_xs_rnn")

    run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=False,
                     wrt_time=split_wrt_time, rnn_type="LSTM", z_dim=10, z_activation="tanh", early_stopping=False,
                     n_epochs=50, batch_size=10, verbose=1,
                     model_name="baseline_noxs_lstm")
    run_baseline_rnn(train_seqs, val_seqs, xs_train, xs_val, vocab, max_seq_length=max_seq_length, orig_seqs_lengths=None, with_xs=True,
                     wrt_time=split_wrt_time, rnn_type="LSTM", z_dim=10, early_stopping=False, n_epochs=50,
                     batch_size=10, verbose=1,
                     model_name="baseline_xs_lstm")

    # baseline = run_rnn_only_ytoy(train_seqs, val_seqs, xs_train, xs_val, vocab, orig_seqs_lengths=None,
    #                              wrt_time=split_wrt_time,
    #                              early_stopping=False, n_epochs=60, batch_size=10,
    #                              verbose=0, model_name="only_y_to_y_model", connect_x=True, y_bias=False,
    #                              connect_y=True, embed_y=False, z_dim=10)

    # run_rnn_xtoy(train_seqs, val_seqs, xs_train, xs_val, vocab, orig_seqs_lengths=None,
    #                   wrt_time=split_wrt_time, rnn_type="simpleRNN", early_stopping=False, n_epochs=80, verbose=0,  model_name="xs_to_y")

    print
    #weights_list=get_model_weights(baseline, "y_to_z_output")
    # print weights_list[0][1]
    # exp_weights=np.exp(weights_list[0][0])
    # n_weights=exp_weights/np.sum(exp_weights,axis=1)
    # print
    # with open('weights_'+"y_to_z_output"+'_.csv', 'wb') as f:
    #     np.savetxt(f, weights_list[0][0], delimiter=',', fmt='%.3f')
    # with open('weights_'+"y_to_y_output"+'.csv', 'wb') as f:
    #     np.savetxt(f, n_weights, delimiter=',', fmt='%.3f')


if __name__ == "__main__":
    # load flickr data
    print "Experimenting with Flickr:"
    seqs, vocab = datasets.load_flickr_data()
    print "total vocab: ", len(vocab)
    # run experiments with flickr real and simulated data
    seqs = datasets.remove_short_seqs(seqs, min_seq_length=2)
    run_experiments(seqs, vocab)
    #run_experiments_on_simulated_data(seqs, vocab, n_seqs=1000)
    print "Experimenting with Flickr ended."

    # print "Experimenting with MSNBC:"
    # seqs, vocab = datasets.load_msnbc_data()
    # print "total vocab: ", len(vocab)
    # # run experiments with MSNBC real and simulated data
    # seqs = datasets.remove_short_seqs(seqs, min_seq_length=2)
    # run_experiments(seqs, vocab)
    # # run_experiments_on_simulated_data(seqs, vocab, n_seqs=1000)
    # print "Experimenting with MSNBC ended."
