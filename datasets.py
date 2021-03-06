"""
Datasets parsing and splitting into train/test sets
"""
import pandas as pd
import numpy as np
import random


def remove_short_seqs(seqs, min_seq_length=1):
    clean_seqs=[]
    for seq in seqs:
        if len(seq)<min_seq_length:
            continue
        clean_seqs.append(seq)
    return clean_seqs


def get_seq_percent(seq, percent, from_start=True):
    seq_length = len(seq)
    if from_start:
        elements = int(np.ceil(percent * seq_length))
        return seq[:elements]
    else:
        elements = int(np.floor(percent * seq_length))
        return seq[seq_length - elements:seq_length]


def split_seqs_wrt_time(sequences, xs, train=0.7, val=0.3, test=0.0):
    """Split list of lists into train/val/test. The validations/tests are taken from the end of each sequence.
            """
    assert train + val + test == 1.0, "ERROR: percents should sum to 1.0"
    train_seqs = []
    val_seqs = []
    test_seqs = []
    xs_train = []
    xs_val = []
    xs_test = []
    for i, seq in enumerate(sequences):
        train_seq = get_seq_percent(seq, train, from_start=True)
        train_x = get_seq_percent(xs[i], train, from_start=True)
        if len(train_seq) > 0:
            train_seqs.append(train_seq)
            xs_train.append(train_x)
        val_test_seq = get_seq_percent(seq, val + test, from_start=False)
        val_test_xs = get_seq_percent(xs[i], val + test, from_start=False)
        val_seq = get_seq_percent(val_test_seq, val / (val + test), from_start=True)
        val_x = get_seq_percent(val_test_xs, val / (val + test), from_start=True)

        if len(val_seq) > 0:
            val_seqs.append(val_seq)
            xs_val.append(val_x)
        test_seq = get_seq_percent(val_test_seq, test / (val + test), from_start=False)
        test_x = get_seq_percent(val_test_xs, test / (val + test), from_start=False)
        if len(test_seq) > 0:
            test_seqs.append(test_seq)
            xs_test.append(test_x)
    return train_seqs, val_seqs, test_seqs, xs_train, xs_val, xs_test


def split_seqs(sequences, shuffle=True, train=0.7, val=0.3, test=0.0):
    """ split list of lists into train/val/test.
    """
    seqs_size = len(sequences)
    indices = range(seqs_size)
    if shuffle:
        random.shuffle(indices)

    train_elems = int(np.ceil(train * seqs_size))
    val_elems = int(np.floor(val * seqs_size))
    train_idxs = indices[:train_elems]
    val_idxs = indices[train_elems:train_elems + val_elems]
    test_idxs = indices[train_elems + val_elems:]

    train_seqs = [sequences[i] for i in train_idxs]
    val_seqs = [sequences[i] for i in val_idxs]
    test_seqs = [sequences[i] for i in test_idxs]
    return train_seqs, val_seqs, test_seqs


def make_splits(sequences, shuffle=True, train=0.7, val=0.3, test=0.0):
    """ split list of lists into train/val/test.
        """
    assert train + val + test == 1.0, "ERROR: percents should sum to 1.0"
    seqs_size = len(sequences)
    indices = range(seqs_size)
    if shuffle:
        random.shuffle(indices)

    train_elems = int(np.ceil(train * seqs_size))
    val_elems = int(np.floor(val * seqs_size))
    train_idxs = indices[:train_elems]
    val_idxs = indices[train_elems:train_elems + val_elems]
    test_idxs = indices[train_elems + val_elems:]

    return train_idxs,val_idxs,test_idxs

def build_xs(sequences, vocab, freq=False):
    """ Constract x's for each sequence.
        args:
            sequences: list of lists.
    """
    xs = []
    for i,seq in enumerate(sequences):
        xi_s = []
        xi = [0] * len(vocab)
        for s in seq:
            if freq:
                xi[s] += 1
            else:
                xi[s] = 1
            xi_s.append(xi[:])
        xs.append(xi_s)
    return xs

def build_seqs_from_df(df, group_col, seq_col, sort_col):
    sorted = df.sort_values([group_col, sort_col], ascending=True)
    grouped = sorted.groupby([group_col])
    seqs = grouped[seq_col].apply(list).values
    return seqs


def load_flickr_data_df():
    """Load Flickr dataset.
        returns:
            data_table: pandas DataFrame.
            stats: pandas DataFrame with dataset statistics
        """
    df = pd.read_csv("data/flickr-data.csv", delimiter=",", parse_dates=[0])
    unique_vals = df["poiID"].unique()
    vocab = dict(zip(unique_vals, range(len(unique_vals))))
    df["poiID"].replace(vocab, inplace=True)
    seqs = build_seqs_from_df(df, "userID", "poiID", "startTime")
    return seqs, vocab


def load_flickr_data(eliminate_repeats=False, gap_thresh=30):
    """Loads flickr dataset.

    args:
        gap_thresh: Duration of time (in minutes) elapsed between events to
            consider next event start of a new sequence.

    returns:
        seqs, vocab
    """
    from datetime import datetime

    vocab_id = 1
    vocab = {'gap': 0} # 0 is the gap token

    seqs = []
    active_uid = None

    with open('data/flickr-data.csv', 'r') as f:
        for i, line in enumerate(f):

            # Skip header
            if i==0:
                continue

            # Parse data
            line = line.rstrip('\r\n')
            vals = line.split(',')
            uid = vals[0]
            token = vals[2]
            curr_start = datetime.fromtimestamp(int(vals[3]))
            curr_end = datetime.fromtimestamp(int(vals[4]))

            if uid != active_uid: # New user logic
                try:
                    seqs.append(active_seq)
                except UnboundLocalError:
                    pass
                active_seq = []
                active_uid = uid
                prev_end = curr_end
                prev_token = None

            # Handle gaps
            time_diff = (curr_start - prev_end).total_seconds() / 60
            if time_diff > gap_thresh:
                active_seq.append(vocab['gap'])

            if token not in vocab:
                vocab[token] = vocab_id
                vocab_id += 1

            if eliminate_repeats and token != prev_token:
                active_seq.append(vocab[token])
                prev_token = token
            elif not eliminate_repeats:
                active_seq.append(vocab[token])

            prev_end = curr_end

    return seqs, vocab


def load_msnbc_data(eliminate_repeats=False):
    """Loads MSNBC dataset.

    Note: This data comes in the form of a list of sequences, so no
    intermediary pandas df is generated.

    returns:
        seqs: A list of sequences.
    """
    vocab_id = 0
    vocab = dict()
    seqs = []
    with open('data/msnbc-data.txt', 'r') as f:
        for i, line in enumerate(f):
            if i>7:
                active_seq = []
                tokens = line.split()
                prev_token = None
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = vocab_id
                        vocab_id += 1
                    if eliminate_repeats and token != prev_token:
                        active_seq.append(vocab[token])
                        prev_token = token
                    elif not eliminate_repeats:
                        active_seq.append(vocab[token])
                seqs.append(active_seq)
        return seqs, vocab


def load_gowalla_data(n_seq=None, bounding_box=None, eliminate_repeats=False):
    """Loads Gowalla dataset.

    Note: This dataset is typically too large to store in memory. Use the
    'n_seqs' parameters to work with a subset of the data.

    args:
        n_seq: Number of sequences to load.
        bounding_box: Bounding box for lat/long coordinates.

    returns:
        seqs: A list of sequences.
    """
    vocab_id = 0
    vocab = dict()

    seqs = []
    active_uid = None
    seq_count = 0

    if bounding_box is None:
        contained = True

    with open('data/gowalla-data.txt', 'r') as f:
        for line in f:
            # Parse data
            vals = line.split('\t')
            uid, token = vals[0], vals[-1]
            lat, lon = float(vals[2]), float(vals[3])

            # Check if enough data has been read
            if seq_count > n_seq:
                break

            # Handle bounding boxes
            if bounding_box is not None:
                contained_lat = (bounding_box.lat[0] < lat) and (lat < bounding_box.lat[1])
                contained_lon = (bounding_box.lon[0] < lon) and (lon < bounding_box.lon[1])
                contained = contained_lat and contained_lon

            # Add data to sequence
            if contained:
                if uid != active_uid: # New user logic
                    try:
                        seqs.append(active_seq)
                    except UnboundLocalError:
                        pass
                    active_seq = []
                    active_uid = uid
                    seq_count += 1
                    prev_token = None
                if token not in vocab:
                    vocab[token] = vocab_id
                    vocab_id += 1
                if eliminate_repeats and token != prev_token:
                    active_seq.append(vocab[token])
                    prev_token = token
                elif not eliminate_repeats:
                    active_seq.append(vocab[token])
    return seqs, vocab


def load_student_data(eliminate_repeats=False, gap_thresh=30):
    """Loads student activity dataset.

    args:
        gap_thresh: Duration of time (in minutes) elapsed between events to
            consider next event start of a new sequence.

    returns:
        seqs, vocab
    """
    from dateutil.parser import parse

    vocab_id = 1
    vocab = {'gap': 0} # 0 is the gap token
    seqs = []
    active_uid = None
    with open('data/student-data.txt', 'r') as f:
        for line in f:
            # Parse data
            line = line.rstrip('\r\n')
            vals = line.split(',')
            uid, dt, ts, token = vals
            curr_datetime = parse(dt+'T'+ts+'Z')
            if uid != active_uid: # New user logic
                try:
                    seqs.append(active_seq)
                except UnboundLocalError:
                    pass
                active_seq = []
                active_uid = uid
                prev_datetime = curr_datetime
                prev_token = None
            # Handle gaps
            time_diff = (curr_datetime - prev_datetime).seconds / 60
            if time_diff > gap_thresh:
                active_seq.append(vocab['gap'])
            if token not in vocab:
                vocab[token] = vocab_id
                vocab_id += 1
            if eliminate_repeats and token != prev_token:
                active_seq.append(vocab[token])
                prev_token = token
            elif not eliminate_repeats:
                active_seq.append(vocab[token])
            prev_datetime = curr_datetime
    return seqs, vocab


def load_reddit_data(eliminate_repeats=False):
    # Initialize
    vocab_id = 0
    vocab = dict()
    seqs = []
    active_uid = None
    with open('data/reddit-data.csv', 'r') as f:
        for line in f:
            vals = line.split(',')
            uid, token = vals[0], vals[1]
            # Start new sequence on new uid
            if uid != active_uid:
                try:
                    seqs.append(active_seq)
                except UnboundLocalError:
                    pass
                active_seq = []
                active_uid = uid
                prev_token = None
            # Logic applied to all lines
            if token not in vocab:
                vocab[token] = vocab_id
                vocab_id += 1
            if eliminate_repeats and token != prev_token:
                active_seq.append(vocab[token])
                prev_token = token
            elif not eliminate_repeats:
                active_seq.append(vocab[token])
    return seqs, vocab


def reddit_generator(filename, batch_size=100, n_seqs=100,with_xs=True,
                     with_x=True):
    from keras.utils import np_utils
    from keras.preprocessing.sequence import pad_sequences
    import cPickle
    from datasets import build_xs
    with open('vocabs/reddit-vocab.pkl', 'rb') as pkl_file:
        vocab = cPickle.load(pkl_file)
    counter = 0
    seqs = []
    while 1:
        with open(filename, 'r') as f:
            l=0
            for i, line in enumerate(f):
                print i,l
                l+=1
                seqs.append(map(int, line.split(",")))
                counter += 1
                if i==n_seqs-1:
                    counter = batch_size
                if counter == batch_size:
                    # print seqs[0]
                    xs=build_xs(seqs, vocab, freq=False)
                    padded = pad_sequences(seqs, maxlen=5001)
                    padded_xs=pad_sequences(xs, maxlen=5001)
                    y = np_utils.to_categorical(padded, num_classes=len(vocab)).reshape((padded.shape[0], padded.shape[1], 1000))
                    if with_xs and with_x:
                        #x = (np.cumsum(y, axis=1) > 0).astype(np.float64)
                        inputs = [y[:, :-1, :], padded_xs[:, :-1, :]]
                    elif with_xs:
                        inputs = padded_xs[:, :-1, :]
                    elif with_x:
                        inputs = y[:, :-1, :]
                    outputs = y[:, 1:, :]
                    # Reset
                    seqs = []
                    counter = 0
                    yield inputs, outputs


def load_switchboard_data():
    import os
    import csv

    data_dir = 'data/swda/'
    vocab_id = 0
    vocab = dict()

    seqs = []
    oldtag2newtag=load_tags_mapping()
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for csv_fname in os.listdir(folder_path):
            seq=[]
            fpath = os.path.join(data_dir, folder, csv_fname)
            if fpath.endswith("csv"):
                with open(fpath, 'r') as f:
                    reader = csv.reader(f)
                    header_ = reader.next()
                    prevtag=""
                    for line in reader:
                        if "@" not in line[4]:
                            for i, col in enumerate(header_):
                                if i < len(line):
                                    if col== 'act_tag':
                                        oldtag=line[i]
                                        newtag = transform_tag(oldtag, oldtag2newtag)
                                        if newtag=="+":
                                            newtag=prevtag
                                        if newtag not in vocab:
                                            vocab[newtag] = vocab_id
                                            vocab_id += 1
                                        seq.append(vocab[newtag])
                                        prevtag = newtag
                                        break
            seqs.append(seq)
    return seqs, vocab


def load_tags_mapping():
    import os
    import csv
    print "Loading label mapping file "
    label_mapping_file = 'data/swbd-damsl_mapping_file.txt'
    if not os.path.exists(label_mapping_file):
        print "ERROR: No label mapping file found!"
        return
    oldtag2newtag = {}
    i = 0
    with open(label_mapping_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            newlab = line[0]
            oldlabs = line[1].split(',')
            i += 1
            for old in oldlabs:
                oldtag2newtag[old] = newlab
    return oldtag2newtag


def transform_tag(oldtag,oldtag2newtag):
    import re
    oldtag = re.sub(r'\**$', '', oldtag)
    if oldtag == '+':  # same as the last one
        return oldtag
    else:
        # remove the trailing * at the end
        newtag = oldtag2newtag.get(oldtag, "")
        if newtag == "":
            if "^" in oldtag:
                ii = 0
                oldtag1 = oldtag
                while newtag == "":
                    oldtag1 = re.sub(r'\(*\^[2gmreqhdtc]\)*$', '', oldtag1)
                    newtag = oldtag2newtag.get(oldtag1, "")
                    ii += 1
                    if ii > 3:
                        break
            if "," in oldtag:
                oldtag2 = oldtag.split(",")[0]
                newtag = oldtag2newtag.get(oldtag2, "")
                if newtag == "":
                    print oldtag, oldtag2
            if ";" in oldtag:
                oldtag2 = oldtag.split(";")[0]
                newtag = oldtag2newtag.get(oldtag2, "")
                if newtag == "":
                    print oldtag, oldtag2

            if newtag == "":
                print oldtag

    return newtag
