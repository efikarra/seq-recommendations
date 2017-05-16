"""Download datasets"""
import gzip
import os
import urllib
from zipfile import ZipFile
import tarfile


def download_gz(url, dest):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with gzip.open(tmpfile, 'rb') as f_in, open(dest, 'wb') as f_out:
        f_out.write(f_in.read())
    os.remove(tmpfile)


def download_zip(url, fname, dest, folder=False):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with ZipFile(tmpfile, 'r') as myzip:
        if folder:
            for f in myzip.namelist():
                myzip.extract(f, dest)
        else:
            with myzip.open(fname, 'r') as f_in, open(dest, 'w') as f_out:
                     f_out.write(f_in.read())
    os.remove(tmpfile)

def download_targz(url, dest_folder):
    tmpfile = 'data.tmp'
    urllib.urlretrieve(url, tmpfile)
    with tarfile.open(tmpfile, 'r:gz') as tarf:
        tarf.extractall(dest_folder)
    os.remove(tmpfile)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', metavar='d', type=str, nargs='+')
    args = parser.parse_args()

    print ''

    if 'flickr' in args.dataset:
        print 'Downloading Flickr dataset'
        url = 'https://raw.githubusercontent.com/arongdari/flickr-photo/master/data/traj-noloop-all-Melb.csv'
        destfile = 'data/flickr-data.csv'
        urllib.urlretrieve(url, destfile)
        print 'Download complete\n'

    if 'gowalla' in args.dataset:
        print 'Downloading Gowalla dataset'
        url = 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
        dest = 'data/gowalla-data.txt'
        download_gz(url, dest)
        print 'Download complete\n'

    if 'msnbc' in args.dataset:
        print 'Downloading MSNBC dataset'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz'
        dest = 'data/msnbc-data.txt'
        download_gz(url, dest)
        print 'Download complete\n'

    if 'student' in args.dataset:
        print 'Downloading student dataset'
        url = 'https://dl.dropboxusercontent.com/u/11521398/student_activity_data.zip'
        fname = 'student_activity_data.csv'
        dest = 'data/student-data.txt'
        download_zip(url, fname, dest)
        print 'Download complete\n'

    if 'reddit' in args.dataset:
        print 'Downloading reddit dataset'
        url = 'https://www.dropbox.com/s/n5yafb040upm16f/reddit-data.tar.gz?dl=1'
        dest = 'data/'
        download_targz(url, dest)
        print 'Download complete\n'

    if 'switchboard' in args.dataset:
        print 'Downloading switchboard dataset'
        url = 'http://compprag.christopherpotts.net/code-data/swda.zip'
        fname = None
        dest = 'data/'
        download_zip(url, fname, dest, folder=True)
        print 'Download complete\n'
