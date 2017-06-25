from __future__ import division
from collections import Counter, defaultdict

c = Counter()
with open('./data/msnbc-data.txt', 'r') as f:
    c.update(f.readlines())
print c.most_common(10)

d = defaultdict(int)
for key, value in c.iteritems():
    length = len(key.split())
    if value > 1 and length > 1:
        d[len(key.split())] += value

d_sum = sum(value for value in d.itervalues())
cum = 0
print d_sum
print "Length\tCount\tCDF"
for key, value in d.iteritems():
    cum += value / d_sum
    print "%i\t%i\t%f" % (key, value, cum)

