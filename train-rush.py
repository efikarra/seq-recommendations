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
    out = seqs_to_array([rw_sampler.gen_sequence(25) for _ in
                         xrange(dataset_size)], vocab = range(k**2))
    return out


if __name__ == '__main__':
    import time
    import cPickle
    import math
    from sampler import RandomWalkSampler
    k = 10
    rw_sampler = RandomWalkSampler(k, betas=[10., 1./10], homeward_bound=False)

    with open('./data/sampler.pkl', 'wb') as pkl:
        cPickle.dump(rw_sampler, pkl)

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

    from keras.callbacks import EarlyStopping

    callback = EarlyStopping(monitor='val_loss', patience=10)
    test = create_samples(10000, 10)

    for dataset_size in [10, 100, 1000, 10000, 100000]:
        train = create_samples(dataset_size, 10)
        dev = create_samples(dataset_size, 10)

        # Markov + X
        name = 'MC-SI'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method='binary',
                            z_dim=None)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        t_start = time.time()
        hist = model.fit(x=train[:,:-1,:],
                         y=train[:,1:,:],
                         batch_size=min([dataset_size, 100]),
                         epochs=1000,
                         validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                         shuffle=True,
                         callbacks=[callback])
        t_end = time.time()
        test_loss = model.test_on_batch(x=test[:,:-1,:], y=test[:,1:,:])
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            line = '%s %i\t%s\n' % (name, dataset_size, test_loss)
            log.write(line)
        with open('data/saved-models/RUSH_time.txt', 'a') as log:
            line = '%s %i\t%0.3f\n' % (name, dataset_size, t_end - t_start)
            log.write(line)

        # Markov + Z=2
        name = 'MC-RNN(2)'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=None,
                            z_dim=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        t_start = time.time()
        hist = model.fit(x=train[:,:-1,:],
                         y=train[:,1:,:],
                         batch_size=min([dataset_size, 100]),
                         epochs=1000,
                         validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                         shuffle=True,
                         callbacks=[callback])
        t_end = time.time()
        test_loss = model.test_on_batch(x=test[:,:-1,:], y=test[:,1:,:])
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            line = '%s %i\t%s\n' % (name, dataset_size, test_loss)
            log.write(line)
        with open('data/saved-models/RUSH_time.txt', 'a') as log:
            line = '%s %i\t%0.3f\n' % (name, dataset_size, t_end - t_start)
            log.write(line)

        # Markov + Z=10
        name = 'MC-RNN(10)'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=None,
                            z_dim=10)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        t_start = time.time()
        hist = model.fit(x=train[:,:-1,:],
                         y=train[:,1:,:],
                         batch_size=min([dataset_size, 100]),
                         epochs=1000,
                         validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                         shuffle=True,
                         callbacks=[callback])
        t_end = time.time()
        test_loss = model.test_on_batch(x=test[:,:-1,:], y=test[:,1:,:])
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            line = '%s %i\t%s\n' % (name, dataset_size, test_loss)
            log.write(line)
        with open('data/saved-models/RUSH_time.txt', 'a') as log:
            line = '%s %i\t%0.3f\n' % (name, dataset_size, t_end - t_start)
            log.write(line)

        # Markov + Z=100
        name = 'MC-RNN(100)'
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=None,
                            z_dim=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        t_start = time.time()
        hist = model.fit(x=train[:,:-1,:],
                         y=train[:,1:,:],
                         batch_size=min([dataset_size, 100]),
                         epochs=1000,
                         validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                         shuffle=True,
                         callbacks=[callback])
        t_end = time.time()
        test_loss = model.evaluate(x=test[:,:-1,:], y=test[:,1:,:],
                                   batch_size=1000)
        model.save_weights('data/saved-models/%s-%i.hdf5' % (name, dataset_size))
        with open('data/saved-models/RUSH_training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s %i\t%s\n' % (name, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/RUSH_valid.txt', 'a') as log:
            line = '%s %i\t%s\n' % (name, dataset_size, test_loss)
            log.write(line)
        with open('data/saved-models/RUSH_time.txt', 'a') as log:
            line = '%s %i\t%0.3f\n' % (name, dataset_size, t_end - t_start)
            log.write(line)

