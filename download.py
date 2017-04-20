"""Download datasets"""
import gzip
import os
import urllib
from zipfile import ZipFile
import tarfile


def download_gz(url, destfile):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with gzip.open(tmpfile, 'rb') as f_in, open(destfile, 'wb') as f_out:
        f_out.write(f_in.read())
    os.remove(tmpfile)


def download_zip(url, fname, destfile):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with ZipFile(tmpfile, 'r') as myzip:
        with myzip.open(fname, 'r') as f_in, open(destfile, 'w') as f_out:
                 f_out.write(f_in.read())
    os.remove(tmpfile)

## TODO all files or just data ???
# def download_tar(url, fname, destfile):
#     tmpfile = 'data.tmp'
#     urllib.urlretrieve(url, tmpfile)
#     with tarfile.open(tmpfile, 'r:gz') as tarf:
#         member = tarf.getmember(fname)
#         tarf.extractall(

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

    if 'student' in args.dataset:
        print 'Downloading student dataset'
        url = 'https://dl.dropboxusercontent.com/u/11521398/student_activity_data.zip'
        fname = 'student_activity_data.csv'
        destfile = 'data/student-data.txt'
        download_zip(url, fname, destfile)
        print 'Download complete\n'

    if 'lastfm' in args.dataset:
        print 'Downloading lastfm dataset'
        url = 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz'
        destfile = 'data/lastfm-data.csv'
        download_gz(url, destfile)
        print 'Download complete\n'

