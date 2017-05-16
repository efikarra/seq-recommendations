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
    train = seqs_to_array([rw_sampler.gen_sequence(25) for _ in xrange(dataset_size)],
                          vocab = range(k**2))
    dev = seqs_to_array([rw_sampler.gen_sequence(25) for _ in xrange(dataset_size)],
                          vocab = range(k**2))
    return train, dev


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('-m', action='store_true')
    parser.add_argument('-z', type=int, default=None)
    parser.add_argument('-x', default=None)
    args = parser.parse_args()

    # print 'Loading train data'
    # with open(args.train) as f:
    #      train = pkl.load(f)

    # print 'Loading test data'
    # with open(args.dev) as f:
    #     dev = pkl.load(f)

    if args.m:
        transition_matrix = generate_transition_matrix(10)
    else:
        transition_matrix = None

    for dataset_size in [10, 100, 1000, 10000, 100000]:
        train, dev = create_samples(dataset_size, 10)
        model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                            transition_matrix=transition_matrix,
                            accumulator_method=args.x,
                            z_dim=args.z)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        hist = model.fit(x=train[:,:-1,:],
                  y=train[:,1:,:],
                  batch_size=min([dataset_size, 100]),
                  epochs=500,
                  validation_data=(dev[:,:-1,:], dev[:,1:,:]),
                  shuffle=True)
        model.save_weights('data/saved-models/%s-%i.hdf5' % (args.id, dataset_size))
        with open('data/saved-models/training_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['loss'])
            line = '%s-%i\t%s\n' % (args.id, dataset_size, loss_seq)
            log.write(line)
        with open('data/saved-models/dev_log.txt', 'a') as log:
            loss_seq = ' '.join(str(loss) for loss in hist.history['val_loss'])
            line = '%s-%i\t%s\n' % (args.id, dataset_size, loss_seq)
            log.write(line)

