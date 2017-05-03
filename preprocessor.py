"""
Preprocessor for keras input format
"""
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np


class Preprocessor():
    def __init__(self, vocab, pad_value=0., seq_length=None):
        self.vocab = vocab
        self.seq_length = seq_length
        self.pad_value = pad_value

    def _pad_sequences(self, sequences, dtype='int32'):
        padded_seqs = pad_sequences(sequences, maxlen=self.seq_length, dtype=dtype, padding='pre', truncating='pre',
                                    value=self.pad_value)
        self.seq_length = padded_seqs.shape[1]
        return padded_seqs.tolist()

    def transform_data(self, sequences, xs=None, pad=True):
        pass


class BaselinePreprocessor(Preprocessor):
    def __init__(self, vocab, pad_value=0., seq_length=None):
        Preprocessor.__init__(self, vocab, pad_value, seq_length)

    def transform_data(self, sequences, xs=None, pad=True):
        x_data = []
        y_data = []
        index = 0
        for seq in sequences:
            x_seq = []
            y_seq = []
            for i in range(0, len(seq) - 1, 1):
                features = []
                targets = []
                features += np_utils.to_categorical([seq[i]], len(self.vocab))[0, :].tolist()
                targets += np_utils.to_categorical([seq[i + 1]], len(self.vocab))[0, :].tolist()
                if xs is not None:
                    features += xs[index][i]
                x_seq.append(features)
                y_seq.append(targets)
            if len(x_seq) > 0:
                x_data.append(x_seq)
                y_data.append(y_seq)
            index += 1
        features_dim = len(self.vocab)
        if xs is not None:
            features_dim += len(self.vocab)
        if pad:
            padded_x_data = self._pad_sequences(x_data, dtype='float64')
            padded_y_data = self._pad_sequences(y_data, dtype='float64')
            padded_y_data = np.reshape(padded_y_data, (len(padded_y_data), self.seq_length, len(self.vocab)))
            padded_x_data = np.reshape(padded_x_data, (len(padded_x_data), self.seq_length, features_dim))
            return padded_x_data, padded_y_data
        else:
            return x_data, y_data


class FullModelPreprocessor(Preprocessor):
    def __init__(self, vocab, pad_value=0., seq_length=None):
        Preprocessor.__init__(self, vocab, pad_value, seq_length)

    def transform_data(self, sequences, xs, pad=True):
        # sequences=sequences[:]
        x_data = []
        y_data = []
        c_data = [x[:-1] for x in xs]
        for j, seq in enumerate(sequences):
            x_seq = []
            y_seq = []
            for i in range(0, len(seq) - 1, 1):
                x_seq.append(np_utils.to_categorical([seq[i]], len(self.vocab))[0, :].tolist())
                y_seq.append(np_utils.to_categorical([seq[i + 1]], len(self.vocab))[0, :].tolist())
            x_data.append(x_seq)
            y_data.append(y_seq)
        features_dim = len(self.vocab)
        padded_x_data = self._pad_sequences(x_data, dtype='float64')
        padded_c_data = self._pad_sequences(c_data, dtype='float64')
        padded_y_data = self._pad_sequences(y_data, dtype='float64')
        padded_y_data = np.reshape(padded_y_data, (len(padded_y_data), self.seq_length, len(self.vocab)))
        padded_c_data = np.reshape(padded_c_data, (len(padded_c_data), self.seq_length, features_dim))
        padded_x_data = np.reshape(padded_x_data, (len(padded_x_data), self.seq_length, features_dim))
        return padded_x_data, padded_y_data, padded_c_data


    def gen_data(self, sequences, xs, with_xs=True,with_x=True,batch_size=100):
        # sequences=sequences[:]

        features_dim = len(self.vocab)
        while True:
            x_data = []
            y_data = []
            c_data=[]
            for j in range(batch_size):
                seq = sequences[j]
                c_data.append(xs[j][:-1])
                x_seq = []
                y_seq = []
                for i in range(0, len(seq) - 1, 1):
                    x_seq.append(np_utils.to_categorical([seq[i]], len(self.vocab))[0, :].tolist())
                    y_seq.append(np_utils.to_categorical([seq[i + 1]], len(self.vocab))[0, :].tolist())
                    x_data.append(x_seq)
                    y_data.append(y_seq)
            padded_x_data = self._pad_sequences(x_data, dtype='float64')
            padded_c_data = self._pad_sequences(c_data, dtype='float64')
            padded_y_data = self._pad_sequences(y_data, dtype='float64')
            padded_y_data = np.reshape(padded_y_data, (len(padded_y_data), self.seq_length, len(self.vocab)))
            padded_c_data = np.reshape(padded_c_data, (len(padded_c_data), self.seq_length, features_dim))
            padded_x_data = np.reshape(padded_x_data, (len(padded_x_data), self.seq_length, features_dim))
            if with_xs and with_x:
                train=[padded_x_data,padded_c_data]
                yield train,padded_y_data
            elif with_xs:
                yield padded_c_data,padded_y_data
            elif with_x:
                yield padded_x_data,padded_y_data



