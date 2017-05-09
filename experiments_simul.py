from sampler import MCSampler,RandomWalkSampler
import datasets
from experiments_methods import *
from model import gauss_prior,ArrayInitializer
from keras.initializers import RandomNormal
from keras.regularizers import l2


def run_experiments_on_simulated_data(seqs, vocab, n_seqs=1000):
    betas = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0]
    betas = [0.2]
    k = 0.000001
    for i, beta in enumerate(betas):
        print
        print
        print "beta: ", beta
        alpha, gamma = utils.transition_matrix(seqs, len(vocab), k=k, freq=False, end_state=False)
        sampler = MCSampler(alpha, gamma, beta=beta, use_end_token=False)
        seqs = []
        for _ in xrange(n_seqs):
            seqs.append(sampler.gen_sequence(n=60))
        seqs = datasets.remove_short_seqs(seqs, min_seq_length=2)
        #utils.rank_plot(seqs, vocab, title="sim data from flickr", save_path="sim_flickr_data")
        max_seq_length = utils.compute_seq_max_length(seqs) - 1
        split_wrt_time = False
        train_seqs, val_seqs, test_seqs = datasets.split_seqs(seqs, shuffle=True, train=0.8, val=0.2, test=0.0)

        freqs = utils.multinomial_probabilities(seqs, len(vocab), k=1.0, normalize=False)
        with open('results/freqs_%d'%i + '.csv', 'wb') as f:
            np.savetxt(f, alpha, delimiter=',', fmt='%.3f')

        train_alpha, train_gamma = utils.transition_matrix(train_seqs, len(vocab), k=k, freq=False, end_state=False)
        init_weights = utils.sample_weights(np.log(train_alpha),np.sqrt(0.1))

        xs_train = datasets.build_xs(train_seqs, vocab, freq=False)
        xs_val = datasets.build_xs(val_seqs, vocab, freq=False)
        print "total train sequences:", len(train_seqs)
        print "total val sequences:", len(val_seqs)
        print "total val sequences:", len(test_seqs)
        mult_model = run_multinomial(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time, normalize=True, k=0.00001)
        markov_model = run_markov(train_seqs, val_seqs, len(vocab), wrt_time=split_wrt_time,k=0.00001)

        n_epochs = 20
        rnn_type = "simpleRNN"
        z_dim = 17
        batch_size = 50
        x_train, y_train, train_xs, x_val, y_val, val_xs = prepare_model_input(train_seqs, val_seqs, xs_train,
                                                                               xs_val, vocab,
                                                                               max_seq_length=max_seq_length,
                                                                               split_wrt_time=split_wrt_time)
        print "ytoy"
        model, results = run_model_no_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                 wrt_time=split_wrt_time, early_stopping=False, n_epochs=n_epochs,
                                                 batch_size=batch_size,
                                                 y_to_y_regularizer=gauss_prior(np.log(train_alpha), 0.001),
                                                 y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                 verbose=1, model_name="ytoy", y_bias=False,
                                                 connect_x=False, connect_y=True)
        weights_list = model.get_layer_weights("x_to_y_output")

        print
        print "ytoz_ytoy"
        model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
                                                   orig_seqs_lengths=None, wrt_time=False, y_to_y_trainable=True,
                                                   y_to_y_w_initializer=ArrayInitializer(init_weights),
                                                   y_to_y_regularizer=gauss_prior(np.log(train_alpha), 0.001),
                                                   read_file=None,
                                                   rnn_type=rnn_type,
                                                   early_stopping=False,
                                                   n_epochs=n_epochs, batch_size=n_epochs, verbose=0, z_dim=z_dim,
                                                   model_name="ytoz_ytoy", y_to_z=True, y_to_y=True,
                                                   x_to_y=False, x_to_z=False, z_to_y_drop=0.1)

        with open("results/ytoz_ytoy" + '_weights_' + '.csv', 'wb') as f:
            np.savetxt(f, weights_list[0], delimiter=',', fmt='%.3f')

        # print
        # print "ytoz"
        # model, results = run_model_with_recurrence(x_train, y_train, train_xs, x_val, y_val, val_xs, vocab,
        #                                            toy_regularizer=None,
        #                                            orig_seqs_lengths=None, wrt_time=False, read_file=None,
        #                                            rnn_type=rnn_type, early_stopping=False,
        #                                            n_epochs=n_epochs, batch_size=batch_size, verbose=0, z_dim=z_dim,
        #                                            model_name="ytoz", y_to_z=True, y_to_y=False,
        #                                            x_to_y=False, x_to_z=False, z_to_y_drop=0.1)
        #



    # alpha, gamma = utils.transition_matrix(train_seqs, len(vocab), k=1.0, freq=False, end_state=False)
    # weights_list = get_model_weights(baseline, "z_to_y_output")
    # exp_weights = np.exp(weights_list[0][0])
    # n_weights = exp_weights / np.sum(exp_weights, axis=1)
    # print
    # with open('weights_' + "alpha" + '_.csv', 'wb') as f:
    #     np.savetxt(f, alpha, delimiter=',', fmt='%.3f')
    # with open('weights_' + "y_to_y_output" + '.csv', 'wb') as f:
    #     np.savetxt(f, n_weights, delimiter=',', fmt='%.3f')
    # run_rnn_fullmodel_xtoy(train_seqs, val_seqs, xs_train, xs_val, vocab, orig_seqs_lengths=None,
    #                   wrt_time=split_wrt_time, rnn_type="simpleRNN", early_stopping=False, n_epochs=80, verbose=0,  model_name="xs_to_y")

    print
    # weights_list=get_model_weights(baseline, "y_to_z_output")
    # print weights_list[0][1]
    # exp_weights=np.exp(weights_list[0][0])
    # n_weights=exp_weights/np.sum(exp_weights,axis=1)
    # print
    # with open('weights_'+"y_to_z_output"+'_.csv', 'wb') as f:
    #     np.savetxt(f, weights_list[0][0], delimiter=',', fmt='%.3f')
    # with open('weights_'+"y_to_y_output"+'.csv', 'wb') as f:
    #     np.savetxt(f, n_weights, delimiter=',', fmt='%.3f')


def run_experiments_on_rw_simulated():
    rw_sampler = RandomWalkSampler(10, 'negative')
    sample_gen = rw_sampler.gen_sample()
    for _ in xrange(10):
        print next(sample_gen)

if __name__ == "__main__":
    run_experiments_on_rw_simulated()
    # load flickr data
    # print "Experimenting with Flickr:"
    # seqs, vocab = datasets.load_flickr_data()
    #seqs, vocab = datasets.load_msnbc_data()
    # import random
    # seqs=[seqs[i] for i in random.sample(xrange(len(seqs)), 20000)]
    #seqs, vocab = datasets.load_reddit_data()
    # BoundingBox = namedtuple('BoundingBox', ['lat', 'lon'])
    # austin_bounds = BoundingBox(
    #     lat=(29.5, 30.5),
    #     lon=(-98.3, -96.9))
    # seqs, vocab = datasets.load_gowalla_data(n_seq=10000, bounding_box=austin_bounds)
    #utils.rank_plot(seqs, vocab, title="msnbc data", save_path="msnbc_data")

    # alpha, gamma = utils.transition_matrix(seqs, len(vocab), k=1.0, freq=False, end_state=False)
    # sampler = MCSampler(alpha, gamma, beta=0.2, use_end_token=False)
    # seqs = []
    # for _ in xrange(1000):
    #     seqs.append(sampler.gen_sequence(n=80))
    # utils.rank_plot(seqs, vocab, title="sim data from flickr", save_path="sim_flickr_data")

    # run experiments with flickr real and simulated data
    # print "total seqs,vocab: ", len(seqs), len(vocab)
    # run_experiments_on_simulated_data(seqs, vocab)
    # print "Experimenting with Flickr ended."
