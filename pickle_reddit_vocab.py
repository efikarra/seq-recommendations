"""Quick hack for fixing reddit vocab"""
import datasets
import cPickle

_, vocab = datasets.load_reddit_data()
with open('data/reddit-vocab.pkl', 'wb') as pkl_file:
    cPickle.dump(vocab, pkl_file)

