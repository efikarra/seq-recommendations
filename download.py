"""Download datasets"""
import gzip
import os
import urllib


def download_gz(url, destfile):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with gzip.open(tmpfile, 'rb') as f_in, open(destfile, 'wb') as f_out:
        f_out.write(f_in.read())
    os.remove(tmpfile)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', metavar='d', type=str, nargs='+')
    args = parser.parse_args()

    print ''

    if 'gowalla' in args.dataset:
        print 'Downloading Gowalla dataset'
        url = 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
        destfile = 'data/gowalla-data.txt'
        download_gz(url, destfile)
        print 'Download complete\n'

    if 'msnbc' in args.dataset:
        print 'Downloading MSNBC dataset'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz'
        destfile = 'data/msnbc-data.txt'
        download_gz(url, destfile)
        print 'Download complete\n'
