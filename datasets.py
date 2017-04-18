"""
@author: efi
"""

import os
import pandas as pd
import numpy as np


def read_data(dir, filename, delimiter=","):
    data_path = os.path.join(dir, filename)
    df = pd.read_csv(data_path, delimiter=delimiter, parse_dates=[0])
    return df


def get_column_vocab(df, col):
    unique_vals = df[col].unique()
    return dict(zip(unique_vals, range(len(unique_vals))))


def remove_short_seqs(df, group_col, seq_col, min_seq_length):
    cleaned = df[df.groupby([group_col])[seq_col].transform(len) > min_seq_length]
    return cleaned


def split_grouped_df(df, group_col, sort_col, seq_col, train=0.7, val=0.3,
                     test=0.0):
    train_df = df.apply(get_group_first_percent, train, seq_col)
    val_test_df = df.apply(get_group_last_percent, val + test, seq_col)
    val_test_sorted = val_test_df.sort_values([group_col, sort_col], ascending=True)
    val_test_grouped = val_test_sorted.groupby([group_col])
    val_df = val_test_grouped.apply(get_group_first_percent, val / (val + test), seq_col)
    test_df = val_test_grouped.apply(get_group_last_percent, test / (val + test), seq_col)
    return train_df, val_df, test_df


def get_group_first_percent(group, percent, col):
    seq_length = len(group[col])
    elements = int(np.ceil(percent * seq_length))
    return group.head(elements)


def get_group_last_percent(group, percent, col):
    seq_length = len(group[col])
    elements = int(np.floor(percent * seq_length))
    return group.tail(elements)


def get_seq_percent(seq, percent, from_start=True):
    seq_length = len(seq)
    if from_start:
        elements = int(np.ceil(percent * seq_length))
        return seq[:elements]
    else:
        elements = int(np.floor(percent * seq_length))
        return seq[seq_length - elements:seq_length]


def data_statistics(df, group_col, seq_col):
    counts_by_user = df.groupby([group_col])[seq_col].count().reset_index()
    basic_stats = pd.DataFrame(
        [counts_by_user[seq_col].min(), counts_by_user[seq_col].max(), counts_by_user[seq_col].median(),
         counts_by_user[seq_col].mean()], \
        index=['min', 'max', 'median', 'mean'])
    basic_stats.columns = ['seq_length']
    return basic_stats


def build_seqs(df, group_col, seq_col, sort_col, vocab=None):
    if vocab:
        df[seq_col].replace(vocab, inplace=True)
    sorted = df.sort_values([group_col, sort_col], ascending=True)
    grouped = sorted.groupby([group_col])
    seqs = grouped[seq_col].apply(list).values
    return seqs


def split_df(df, group_col, seq_col, sort_col, train=0.7, val=0.3, test=0.0,vocab=None):
    """Split data into train, validation and test sequences. The validations/tests are taken from the end of each sequence.
        args:
            data: pandas DataFrame. Data to be splitted.
            train: float. Percent of sequences as training data.
            val: float. Percent of sequences as validation data.
            test: float or None. Percent of sequences as test data. If None, split only into train/val
            min_seq_length: int or None. If int, remove sequences with length less than min_seq_length.
        returns:
            train_table,val_table,test_table: pandas DataFrames.
        """
    assert train + val + test == 1.0, "ERROR: percents should sum to 1.0"
    if vocab:
        df[seq_col].replace(vocab, inplace=True)
    sorted = df.sort_values([group_col, sort_col], ascending=True)
    grouped = sorted.groupby([group_col])
    train_df, val_df, test_df = split_grouped_df(grouped, group_col, sort_col, seq_col,
                                                 train, val, test)
    return train_df, val_df, test_df


def split_seq_data(sequences, vocab, train=0.7, val=0.3, test=0.0):
    assert train + val + test == 1.0, "ERROR: percents should sum to 1.0"
    train_seqs = []
    val_seqs = []
    test_seqs = []
    for seq in sequences:
        train_seq = get_seq_percent(seq, train, from_start=True)
        if len(train_seq) > 0:
            train_seqs.append(train_seq)
        val_test_seq = get_seq_percent(seq, val + test, from_start=False)
        val_seq = get_seq_percent(val_test_seq, val / (val + test), from_start=True)
        if len(val_seq) > 0:
            val_seqs.append(val_seq)
        test_seq = get_seq_percent(val_test_seq, test / (val + test), from_start=False)
        if len(test_seq) > 0:
            test_seqs.append(test_seq)
    xs_train = build_xs(train_seqs, vocab)
    print "total train sequences:", len(train_seqs)

    xs_val = build_xs(val_seqs, vocab)
    print "total val sequences:", len(val_seqs)

    xs_test = build_xs(test_seqs, vocab)
    print "total test sequences:", len(test_seqs)
    return train_seqs, val_seqs, test_seqs, xs_train, xs_val, xs_test


def build_xs(sequences, vocab):
    xs = []
    for seq in sequences:
        xi_s = []
        xi = [0] * len(vocab)
        for s in seq:
            xi[vocab[s]] = 1
            xi_s.append(xi[:])
        xs.append(xi_s)
    return xs


def load_flickr_data():
    """Load Flickr dataset.
        returns:
            data_table: pandas DataFrame.
            stats: pandas DataFrame with dataset statistics
        """
    df = read_data("data/", "traj-noloop-all-Melb.csv", delimiter=",")
    return df


def clean_flickr_data(df, min_seq_length):
    return remove_short_seqs(df, group_col="userID", seq_col="poiID", min_seq_length=min_seq_length)


def flickr_table_statistics(df):
    """Get basic statistics for flickr sequences.
        args:
            data_table: pandas DataFrame.
        returns:
            stats: pandas DataFrame with dataset statistics
        """
    return data_statistics(df, "userID", "poiID")


def build_flickr_seqs(flickr_df, xs=False):
    """Build sequences from Flickr data frame.
        args:
            flickr_data: pandas DataFrame.
            xs: bool. True if you want the x's of the sequences
        returns:
            flickr_seqs: list of lists. sequences.
            vocab: dict. vocabulary of locations
            xs_total: list of lists. x's.
        """
    vocab = get_column_vocab(flickr_df, "poiID")
    flickr_seqs = build_seqs(flickr_df, "userID", "poiID", "startTime",vocab)
    xs_total = None
    if xs:
        xs_total = build_xs(flickr_seqs, vocab)
    return flickr_seqs, vocab, xs_total


def split_flickr_train_val_df(df, train=0.7, val=0.3, test=0.0):
    vocab = get_column_vocab(df, "poiID")
    train_df, val_df, test_df = split_df(df, "userID", "poiID", "startTime", train,
                                         val, test,vocab)
    return train_df, val_df, test_df


def build_flickr_train_val_seqs(train_table, val_table, test_table):
    """Split data into train, validation and test sequences. The validations/tests are taken from the end of each sequence.
        args:
            train_table,val_table,test_table
        returns:
            train_seqs,val_seqs,test_seqs: list of lists. train/val/test sequences
            xs_train,xs_val,xs_test: list of lists. train/val/test x's
            vocab: dict. vocabulary of indices to location ids.
        """
    train_seqs = build_seqs(train_table, "userID", "poiID", "startTime")
    vocab = get_column_vocab(train_table, "poiID")
    xs_train = build_xs(train_seqs, vocab)
    print "train vocabulary size:", len(vocab)
    print "total train sequences:", len(train_seqs)

    val_seqs = build_seqs(val_table, "userID", "poiID", "startTime")
    xs_val = build_xs(val_seqs, vocab)
    print "total val sequences:", len(val_seqs)
    test_seqs = build_seqs(test_table, "userID", "poiID", "startTime")
    xs_test = build_xs(test_seqs, vocab)
    print "total test sequences:", len(test_seqs)
    return train_seqs, val_seqs, test_seqs, xs_train, xs_val, xs_test, vocab


def load_msnbc_data():
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
                seq = []
                vals = line.split()
                for val in vals:
                    if val not in vocab:
                        vocab[val] = vocab_id
                        vocab_id += 1
                    seq.append(vocab[val])
                seqs.append(seq)
        return seqs, vocab


def load_gowalla_data(n_seq=None, bounding_box=None):
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
            uid, loc = vals[0], vals[-1]
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
                if loc not in vocab:
                    vocab[loc] = vocab_id
                    vocab_id += 1
                active_seq.append(vocab[loc])

    return seqs, vocab

