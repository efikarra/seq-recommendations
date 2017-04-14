'''
Created on 12 Apr 2017

@author: efi
'''

import os
import pandas as pd
import numpy as np


def _read_data(data_dir, filename, delimiter=","):
    data_path = os.path.join(data_dir, filename)
    data_table = pd.read_csv(data_path, delimiter=delimiter, parse_dates=[0])
    return data_table


def _group_sort_by_columns(data_table, group_col, sort_col):
    grouped_by_user = data_table.sort_values([group_col, sort_col], ascending=True).groupby([group_col])
    return grouped_by_user


def _get_groups_as_lists(grouped_by_col, column):
    grouped_data = grouped_by_col[column].apply(list).values
    return grouped_data


def _get_column_vocab(data_table, column):
    unique_pois = data_table[column].unique()
    return dict(zip(unique_pois, range(len(unique_pois))))


def _remove_short_sequences(data_table, group_col, seq_col, min_seq_length):
    clean_table = data_table[data_table.groupby([group_col])[seq_col].transform(len) > min_seq_length]
    return clean_table


def _split_grouped_data(grouped_data, group_col, sort_col, column, train_percent=0.7, val_percent=0.3,
                        test_percent=0.0):
    train_table = grouped_data.apply(_get_group_first_percentage, train_percent, column)
    val_test_table = grouped_data.apply(_get_group_last_percentage, val_percent + test_percent, column)

    val_test_table = _group_sort_by_columns(val_test_table, group_col=group_col, sort_col=sort_col)
    val_table = val_test_table.apply(_get_group_first_percentage, val_percent / (val_percent + test_percent), column)
    test_table = val_test_table.apply(_get_group_last_percentage, test_percent / (val_percent + test_percent), column)
    return train_table, val_table, test_table


def _get_group_first_percentage(group, percent, column):
    seq_length = len(group[column])
    elements = int(np.ceil(percent * seq_length))
    return group.head(elements)


def _get_group_last_percentage(group, percent, column):
    seq_length = len(group[column])
    elements = int(np.floor(percent * seq_length))
    return group.tail(elements)


def _get_seq_percentage(seq, percent, from_start=True):
    seq_length = len(seq)
    if from_start:
        elements = int(np.ceil(percent * seq_length))
        return seq[:elements]
    else:
        elements = int(np.floor(percent * seq_length))
        return seq[seq_length - elements:seq_length]


def _data_statistics(data_table, group_col, seq_col):
    counts_by_user = data_table.groupby([group_col])[seq_col].count().reset_index()
    basic_stats = pd.DataFrame(
        [counts_by_user[seq_col].min(), counts_by_user[seq_col].max(), counts_by_user[seq_col].median(),
         counts_by_user[seq_col].mean()], \
        index=['min', 'max', 'median', 'mean'])
    basic_stats.columns = ['seq_length']
    return basic_stats


def _build_sequences(data, group_col, seq_col, sort_col):
    grouped_by_user = _group_sort_by_columns(data, group_col=group_col, sort_col=sort_col)
    flickr_seqs = _get_groups_as_lists(grouped_by_user, seq_col)
    return flickr_seqs


def _split_table_data(data, group_col, seq_col, sort_col, train_percent=0.7, val_percent=0.3, test_percent=0.0):
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
    assert train_percent + val_percent + test_percent == 1.0, "percents should sum to 1.0"
    grouped_by_user = _group_sort_by_columns(data, group_col=group_col, sort_col=sort_col)
    train_table, val_table, test_table = _split_grouped_data(grouped_by_user, group_col, sort_col, seq_col,
                                                             train_percent, val_percent, test_percent=test_percent)
    return train_table, val_table, test_table


def split_seq_data(sequences, vocab, train_percent=0.7, val_percent=0.3, test_percent=0.0):
    assert train_percent + val_percent + test_percent == 1.0, "percents should sum to 1.0"
    train_seqs = []
    val_seqs = []
    test_seqs = []
    for seq in sequences:
        train_seq = _get_seq_percentage(seq, train_percent, from_start=True)
        if len(train_seq) > 0:
            train_seqs.append(train_seq)
        val_test_seq = _get_seq_percentage(seq, val_percent + test_percent, from_start=False)
        val_seq = _get_seq_percentage(val_test_seq, val_percent / (val_percent + test_percent), from_start=True)
        if len(val_seq) > 0:
            val_seqs.append(val_seq)
        test_seq = _get_seq_percentage(val_test_seq, test_percent / (val_percent + test_percent), from_start=False)
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
    data_table = _read_data("../data/", "traj-noloop-all-Melb.csv", delimiter=",")
    return data_table


def clean_flickr_data(data, min_seq_length):
    return _remove_short_sequences(data, group_col="userID", seq_col="poiID", min_seq_length=min_seq_length)


def flickr_table_statistics(data_table):
    """Get basic statistics for flickr sequences.
        args:
            data_table: pandas DataFrame.
        returns: 
            stats: pandas DataFrame with dataset statistics
        """
    return _data_statistics(data_table, "userID", "poiID")


def build_flickr_sequences(flickr_data, xs=False):
    """Build sequences from Flickr data frame.
        args:
            flickr_data: pandas DataFrame.
            xs: bool. True if you want the x's of the sequences
        returns: 
            flickr_seqs: list of lists. sequences.
            vocab: dict. vocabulary of locations
            xs_total: list of lists. x's.
        """
    flickr_seqs = _build_sequences(flickr_data, "userID", "poiID", "startTime")
    vocab = _get_column_vocab(flickr_data, "poiID")
    xs_total = None
    if xs:
        xs_total = build_xs(flickr_seqs, vocab)
    return flickr_seqs, vocab, xs_total


def split_flickr_train_val_table(data, train_percent=0.7, val_percent=0.3, test_percent=0.0):
    train_table, val_table, test_table = _split_table_data(data, "userID", "poiID", "startTime", train_percent,
                                                           val_percent, test_percent)
    return train_table, val_table, test_table


def build_flickr_train_val_seqs(train_table, val_table, test_table):
    """Split data into train, validation and test sequences. The validations/tests are taken from the end of each sequence.
        args:
            train_table,val_table,test_table
        returns:
            train_seqs,val_seqs,test_seqs: list of lists. train/val/test sequences
            xs_train,xs_val,xs_test: list of lists. train/val/test x's
            vocab: dict. vocabulary of indices to location ids.
        """
    train_seqs = _build_sequences(train_table, "userID", "poiID", "startTime")
    vocab = _get_column_vocab(train_table, "poiID")
    xs_train = build_xs(train_seqs, vocab)
    print "train vocabulary size:", len(vocab)
    print "total train sequences:", len(train_seqs)

    val_seqs = _build_sequences(val_table, "userID", "poiID", "startTime")
    xs_val = build_xs(val_seqs, vocab)
    print "total val sequences:", len(val_seqs)
    test_seqs = _build_sequences(test_table, "userID", "poiID", "startTime")
    xs_test = build_xs(test_seqs, vocab)
    print "total test sequences:", len(test_seqs)
    return train_seqs, val_seqs, test_seqs, xs_train, xs_val, xs_test, vocab
