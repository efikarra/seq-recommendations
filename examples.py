"""
@author: 1. Robert, 2. Efi

examples of reading datasets
"""
from __future__ import division

from datasets import load_gowalla_data,load_msnbc_data,load_flickr_data,load_student_data
from collections import namedtuple

### Flickr
print
print "--- Loading Flickr data ---"

flickr_data, vocab = load_flickr_data()

print "Dataset size:", len(flickr_data)
print "Vocab size:", len(vocab)
print "Avg sequence length:", sum(map(lambda x: len(x), flickr_data)) / len(flickr_data)


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

### Student
print
print "--- Loading Student data ---"

student_data, vocab = load_student_data()

def split_gaps(seqs):
    import itertools
    for seq in seqs:
        split = [list(g) for k,g in itertools.groupby(seq, lambda x: x==0) if not k]
        for subseq in split:
            yield(subseq)

split_student_data = list(split_gaps(student_data))

print "Dataset size:", len(student_data)
print "Vocab size:", len(vocab)
print "Avg sequence length (not breaking on gaps):", sum(map(lambda x: len(x), student_data)) / len(student_data)
print "Avg sequence length (breaking on gaps):", sum(map(lambda x: len(x), split_student_data)) / len(split_student_data)
print

