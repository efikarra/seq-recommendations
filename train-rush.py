""" Train model """
import cPickle as pkl
from itertools import product
from model import build_model
import numpy as np


def generate_transition_matrix(k):
    transition_matrix = np.zeros((k**2, k**2))
    span = xrange(k)
    disp = [-1, 0, 1]
    for i, j in product(span, span):
        pos = i*k + j
        for di, dj in product(disp, disp):
            x = i + di
            y = j + dj
            neighb_pos = x*k + y
            test1 = (di != 0) or (dj != 0)
            test2 = (0 <= x < k)
            test3 = (0 <= y < k)
            if test1 and test2 and test3:
                transition_matrix[pos][neighb_pos] = 1
    return transition_matrix


def create_samples(dataset_size, k):
    from datasets import seqs_to_array
    from sampler import RandomWalkSampler
    import math
    rw_sampler = RandomWalkSampler(k, betas=[math.e], homeward_bound=False)
    out = seqs_to_array([rw_sampler.gen_sequence(25) for _ in
                         xrange(dataset_size)], vocab = range(k**2))
    return out


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    # print 'Loading train data'
    # with open(args.train) as f:
    #      train = pkl.load(f)

    # print 'Loading test data'
    # with open(args.dev) as f:
    #     dev = pkl.load(f)

    transition_matrix = generate_transition_matrix(10)

    for dataset_size in [10, 100, 1000, 10000, 100000]:
        dev = create_samples(dataset_size, 10)
        train = create_samples(dataset_size, 10)

        # Markov + X
        name = 'MARKOV-X'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method='count',
                            z_dim=None)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        hist = model.fit(x=train[:,:-1,:],
                  y=train[:,1:,:],
                  batch_size=min([dataset_size, 100]),
                  epochs=500,
                  validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                  shuffle=True)
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            loss_seq = str(hist.history['val_loss'][-1])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)

        # Markov + Z=2
        name = 'MARKOV-Z=2'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=None,
                            z_dim=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        hist = model.fit(x=train[:,:-1,:],
                  y=train[:,1:,:],
                  batch_size=min([dataset_size, 100]),
                  epochs=500,
                  validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                  shuffle=True)
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            loss_seq = str(hist.history['val_loss'][-1])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)

        # Markov + Z=10
        name = 'MARKOV-Z=10'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=None,
                            z_dim=10)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        hist = model.fit(x=train[:,:-1,:],
                  y=train[:,1:,:],
                  batch_size=min([dataset_size, 100]),
                  epochs=500,
                  validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                  shuffle=True)
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            loss_seq = str(hist.history['val_loss'][-1])
            line = '%s-%i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)

