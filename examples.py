"""
@author: efi

examples of reading datasets
"""
from __future__ import division

from datasets import *
from collections import namedtuple

# load flickr DataFrame
flickr_data = load_flickr_data()
# get statistics for the whole dataset
print "all data statistics:"
print flickr_table_statistics(flickr_data)
print
# remove flickr sequences (i.e users) with length less than min_seq_length (if you want).
flickr_data = clean_flickr_data(flickr_data, min_seq_length=1)

# 1. You can directly split the flickr data frame into train/val/test data.
# Set test_percent=0.0 if you only want to split into train/val sets.
train_table, val_table, test_table = split_flickr_train_val_df(flickr_data, train=0.7, val=0.3,
                                                               test=0.0)
seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test, vocab = build_flickr_train_val_seqs(train_table, val_table,
                                                                                                test_table)
# get statistics for the train/test sets
print
print "train statistics:"
print flickr_table_statistics(train_table)
print
print "val statistics:"
print flickr_table_statistics(val_table)
print

# 2. Or you can extract the whole sequences from flickr data frame and then you can split the sequences into
# train/val/test sequences. If xs=True you get the xs for each sequence before splitting into train/val/test.
sequences, vocab, total_xs = build_flickr_seqs(flickr_data, xs=True)
seqs_train, seqs_val, seqs_test, xs_train, xs_val, xs_test = split_seq_data(sequences, vocab, train=0.7,
                                                                            val=0.3, test=0.0)
### MSNBC
print
print "--- Loading MSNBC data ---"

msnbc_data, vocab = load_msnbc_data()

print "Dataset size:", len(msnbc_data)
print "Vocab size:", len(vocab)
print "Avg sequence length:", sum(map(lambda x: len(x), msnbc_data)) / len(msnbc_data)

### Gowalla
print
print "--- Loading Gowalla data in Austin ---"

BoundingBox = namedtuple('BoundingBox', ['lat', 'lon'])
austin_bounds = BoundingBox(
    lat=(29.5, 30.5),
    lon=(-98.3, -96.9))
gowalla_data, vocab = load_gowalla_data(n_seq=10000, bounding_box=austin_bounds)

print "Dataset size:", len(gowalla_data)
print "Vocab size:", len(vocab)
print "Avg sequence length:", sum(map(lambda x: len(x), gowalla_data)) / len(gowalla_data)
print

